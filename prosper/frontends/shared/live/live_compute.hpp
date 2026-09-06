#pragma once
#include "gpu/execute/gpu_execute.hpp"
#include "shared/compute/compute_image_borrow_census.hpp"
#include "shared/texture/write_watch_census.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace prosper::gpu {
struct MipChainPlan;
}

namespace prosper::frontend {

enum class ComputeImageCacheClass : uint8_t { sampled, storage };

// Persistent SSBO identity must distinguish guest bytes from their Vulkan materialization. A
// one-record 16-bit source and an ordinary four-byte source can otherwise share address, host-data
// identity, and bound size while requiring different upper bytes and typed shader semantics.
struct ComputeBufferMaterializationDiscriminator {
    uint64_t logical_bytes = 0;
    uint64_t binding_bytes = 0;
    prosper::gpu::StorageBufferTailSemantic semantic =
        prosper::gpu::StorageBufferTailSemantic::None;

    bool operator==(const ComputeBufferMaterializationDiscriminator&) const = default;
};

constexpr ComputeBufferMaterializationDiscriminator
compute_buffer_materialization_discriminator(
    const prosper::gpu::StorageBufferMaterializationPlan& plan) {
    return {plan.logical_bytes, plan.binding_bytes, plan.semantic};
}

// One 4 KiB host page is the measured crossover for both storage and sampled images, avoiding
// retaining sub-page Vulkan objects while eliminating repetitive CPU staging allocation, CPU
// detiling, and Vulkan image creation for sub-1MB textures (#3291).
constexpr uint64_t compute_image_cache_default_minimum_bytes(
    ComputeImageCacheClass image_class) {
    return 4ull * 1024ull;
}

constexpr bool compute_image_cache_default_eligible(
    uint64_t bytes, ComputeImageCacheClass image_class) {
    return bytes >= compute_image_cache_default_minimum_bytes(image_class);
}

// Large portable-CFG modules are intentionally explicit: one guest instruction can become several
// SPIR-V operations plus dispatcher state. Keep the 32K-word classification independently testable,
// but optimize these modules by default. Disabling driver optimization reduced cold compile latency
// on NVIDIA at the cost of turning GTA V's repeated 1440p dispatches from roughly 1.7 ms into 47-50
// ms. PROSPER_COMPUTE_FAST_COMPILE_LARGE remains the diagnostic opt-out for investigations where
// bounded first-use latency matters more than steady-state execution.
constexpr bool compute_pipeline_is_large(size_t spirv_words) {
    return spirv_words >= 32u * 1024u;
}

// Vulkan pipeline-cache blobs are driver-owned, but their fixed version-one prefix identifies the
// exact vendor, device, and pipelineCacheUUID that may consume them. Validate that prefix ourselves
// before passing persistent bytes to vkCreatePipelineCache: a stale driver cache is then a clean
// miss rather than a startup failure. The UUID is passed as bytes to keep Vulkan types out of this
// shared policy header and its unit tests.
bool compute_pipeline_cache_blob_compatible(
    const uint8_t* blob, size_t blob_bytes,
    uint32_t vendor_id, uint32_t device_id,
    const uint8_t* pipeline_cache_uuid, size_t uuid_bytes);

// A sampled descriptor needs guest validation/conversion only when neither renderer authority path
// supplied its pixels. Depth images can be imported from the persistent Vulkan DS cache even though
// they deliberately have no color-RTT identity; that import must not fall through and snapshot stale
// guest backing that the dispatch will never consume.
constexpr bool compute_sampled_guest_prepare_required(bool storage_image,
                                                       bool renderer_owned,
                                                       bool imported_image,
                                                       bool imported_bypass_disabled = false) {
    return !storage_image && !renderer_owned &&
           (!imported_image || imported_bypass_disabled);
}

// #3407. `compute_sampled_guest_prepare_required` above excludes renderer-owned surfaces, and
// rightly so: that path resolves and validates GUEST bytes, and a surface prosper's own renderer
// produced has none. The consequence was that such a surface had no cache identity at all, so every
// dispatch reconverted it -- measured on Sonic Frontiers' intro at 441 conversions of a single
// 33 MB plane against 148 distinct published buffers.
//
// The identity it does have is the publication: `LiveTargetSnapshot::pixels` is a
// `shared_ptr<const std::vector<uint8_t>>`. A cache entry that HOLDS that shared_ptr keeps the
// buffer alive (so its address cannot be reused) and the buffer is const (so its bytes cannot
// change), which is what makes pointer equality a sound proof of byte equality. Both halves are
// load-bearing: holding a raw address instead would admit an ABA match after the renderer frees and
// reallocates, and that would bind a stale image -- silent visual corruption, the failure this
// project least wants.
struct ComputeRendererConversionCacheInputs {
    bool renderer_owned = false;
    bool storage_image = false;
    bool imported = false;
    bool depth_bits_source = false;
    bool seed_skip = false;
    bool compute_transfer_seed_borrowed = false;
    bool seeded_from_imported = false;
    bool has_published_pixels = false;
    // The conversion chain treats a renderer publication as one whole 2D surface. Volume, array and
    // mip-chain shapes are excluded rather than reasoned about; the cost this exists for is 2D 4K
    // planes, and a wrong hit here is silent visual corruption rather than a slow frame.
    bool volume_or_array = false;
    uint32_t array_layers = 1;
    uint32_t mip_levels = 1;
    bool cache_enabled = true;
    bool persistent_enabled = false;
};

constexpr bool compute_renderer_conversion_cache_candidate(
        const ComputeRendererConversionCacheInputs& in) {
    return in.cache_enabled && in.persistent_enabled &&
           in.renderer_owned && in.has_published_pixels &&
           !in.storage_image && !in.imported && !in.depth_bits_source &&
           !in.seed_skip && !in.compute_transfer_seed_borrowed && !in.seeded_from_imported &&
           !in.volume_or_array && in.array_layers == 1u && in.mip_levels == 1u;
}

// The whole correctness argument for the cache, in one predicate so it can be tested with a
// mutation arm per clause rather than inferred from the call site. `content_valid` is false while a
// transfer is outstanding or after a failed submit, so a hit can never be authorized against an
// image that did not receive these bytes.
constexpr bool compute_renderer_conversion_cache_hit(bool content_valid,
                                                      const void* cached_publication,
                                                      const void* current_publication) {
    return content_valid && cached_publication != nullptr &&
           current_publication != nullptr &&
           cached_publication == current_publication;
}

// Materialize a proven uniform DCC fast clear for the narrow sampled representation used by an
// ordinary guest-backed 2D RGBA16F compute input. The complete metadata plane, guest shape,
// reflected non-arrayed view, and rollback state are part of the proof; any unsupported state
// returns false and leaves the caller on its established base-byte path. `texel_count` may be one
// for eligibility probing or the complete image size for staging materialization.
bool compute_sampled_dcc_fast_clear_rgba8(
    const prosper::gpu::ShaderResource& resource,
    bool ordinary_guest_backed_sampled_view,
    bool arrayed_sampled_view,
    bool disabled,
    uint8_t* rgba,
    size_t texel_count,
    const uint8_t* metadata,
    size_t metadata_bytes,
    uint8_t* clear_code = nullptr);

// Keep enough persistent image residency for modern multi-pass workloads without claiming an
// unreasonable share of small discrete GPUs. The 512 MiB historical floor remains appropriate for
// a 4 GiB heap, while larger devices contribute one eighth of their local heap up to 2 GiB. An
// explicit PROSPER_COMPUTE_IMAGE_CACHE_MB value still overrides this production default.
constexpr uint64_t compute_image_cache_default_limit_bytes(uint64_t device_local_bytes) {
    constexpr uint64_t floor = 512ull * 1024ull * 1024ull;
    constexpr uint64_t ceiling = 2048ull * 1024ull * 1024ull;
    const uint64_t scaled = device_local_bytes / 8ull;
    return scaled < floor ? floor : (scaled > ceiling ? ceiling : scaled);
}

// Pack one raw float32 channel to UNORM8 using the storage-image conversion contract. Kept public
// so the optimized scalar conversion can be checked directly against the previous lround path.
uint8_t storage_pack_unorm8(uint32_t float_bits);
uint16_t storage_pack_unorm16(uint32_t float_bits);
void storage_pack_unorm8_range(const uint32_t* channels, uint32_t components,
                               size_t texels, uint8_t* packed);

// Hot storage/sampled-image conversions. These retain the scalar reference semantics while using
// exhaustive binary16 lookup tables or exact runtime-dispatched vector paths in production loops.
uint32_t storage_unpack_float16_bits(uint16_t half_bits);
// Expand tightly-strided RGBA16F to RGBA32F channel bits. The F16C path repairs signaling-NaN
// payloads so every output bit remains identical to half_to_float.
void storage_unpack_float16x4_range(const uint8_t* rgba16f, size_t texels, uint32_t* channels);
uint8_t sampled_float16_to_unorm8(uint16_t half_bits);
void sampled_float16_to_unorm8_range(const uint8_t* source, uint32_t components,
                                     size_t texels, uint8_t* rgba);
// Pack tightly-strided RGBA32F channel bits to RGBA16F. Uses a runtime-dispatched F16C path where
// available and preserves float_to_half's exact NaN payload/rounding contract.
void storage_pack_float16x4_range(const uint32_t* channels, size_t texels, uint8_t* rgba16f);
// Narrow a Float32 SAMPLED surface to RGBA16F, synthesising the channels the source does not carry
// as the hardware's (0,0,0,1). `source_texel_bytes` is the source stride, which is not always
// `components * 4`, and `components` above four are ignored the way a four-channel destination
// requires.
//
// This exists because the renderer had the same conversion written out as a per-texel, per-channel
// scalar loop around float_to_half with a 4-byte and a 2-byte memcpy per channel -- 8.3 million
// conversions and 16.6 million small copies for one 1920x1080 reference, on one thread, while the
// vectorised sibling above sat beside it. Routing through that sibling picks up both its F16C path
// and its thread split; the bit-for-bit contract is the same one, and is asserted against the
// original scalar loop for every component count in test_game_compute.
void pack_float32_to_rgba16f_range(const uint8_t* source, uint32_t components,
                                   size_t source_texel_bytes, size_t texels,
                                   uint8_t* rgba16f);

// Portable storage-image ABI shared by compute and graphics: guest texels are expanded into four
// raw 32-bit VGPR channel values and bound through an unsigned-integer Vulkan image. The helpers
// reject unknown formats, invalid component counts, and short spans instead of guessing a layout.
uint32_t storage_image_guest_texel_bytes(prosper::gpu::DataFormat format,
                                         uint32_t components);
bool storage_image_unpack_raw_uvec4(const uint8_t* source, size_t source_bytes,
                                    prosper::gpu::DataFormat format, uint32_t components,
                                    size_t texels, uint32_t* channels,
                                    size_t channel_dwords);

// Exact source footprint and complete guest->portable materialization for the shared raw-uvec4
// storage-image ABI. Tiled sources must provide the whole padded allocation: the detilers index that
// footprint without a source-length argument, so accepting only the tight linear byte count would
// silently turn an absent tiled tail into zero texels. Zero/false means unsupported shape, overflow,
// or short backing. `depth` is one for a 2D image and the real slice count for a volume.
size_t storage_image_raw_uvec4_source_bytes(
    prosper::gpu::DataFormat format, uint32_t components,
    uint32_t width, uint32_t height, uint32_t depth, uint32_t tile_mode,
    bool in_mip_tail, uint32_t mip_tail_bytes);
bool storage_image_materialize_raw_uvec4(
    const uint8_t* source, size_t source_bytes,
    prosper::gpu::DataFormat format, uint32_t components,
    uint32_t width, uint32_t height, uint32_t depth, uint32_t tile_mode,
    bool in_mip_tail, uint32_t mip_tail_bytes,
    uint32_t mip_tail_x, uint32_t mip_tail_y,
    uint32_t* channels, size_t channel_dwords);

// Reverse the portable raw-uvec4 materialization after a writable Vulkan storage image has
// completed. `channels` contains four 32-bit values per texel; the helper restores the descriptor's
// compact guest format and then its linear, tiled-surface, mip-tail, or tiled-volume layout. The
// destination must cover the complete footprint returned by storage_image_raw_uvec4_source_bytes.
bool storage_image_writeback_raw_uvec4(
    const uint32_t* channels, size_t channel_dwords,
    prosper::gpu::DataFormat format, uint32_t components,
    uint32_t width, uint32_t height, uint32_t depth, uint32_t tile_mode,
    bool in_mip_tail, uint32_t mip_tail_bytes,
    uint32_t mip_tail_x, uint32_t mip_tail_y,
    uint8_t* destination, size_t destination_bytes);

// A typed Vulkan storage image already exposes the guest format as exact row-major bytes. For a
// tiled guest surface the tiler can therefore read the mapped staging image directly, unless a
// poison-proving dispatch still needs a mutable linear copy to restore untouched texels.
constexpr bool storage_writeback_can_tile_mapped_bytes(bool exact_storage_bytes,
                                                       uint32_t tile_mode,
                                                       bool poison_verify,
                                                       bool disabled) {
    return exact_storage_bytes && tile_mode != 0 && !poison_verify && !disabled;
}

// Whether a sampled guest view can bind a renderer-owned target without a CPU readback/conversion.
// Exact Vulkan formats may always be reused. The sole cross-format case is RGBA16_UNORM backed by
// canonical RGBA8 values, and only when reflection proves that the sampled image returns float
// values. Exact-extent checks remain a separate coordinate contract for texel fetch and queries.
bool direct_sampled_rtt_compatible(prosper::gpu::DataFormat format, uint32_t components,
                                   prosper::gpu::LiveTargetPixelFormat target_format,
                                   bool float_sampled_values);

// Whether a CPU snapshot of a renderer-owned target can be copied byte-for-byte into the sampled
// image format requested by a compute descriptor. Unlike direct binding, this may admit a
// same-width typed alias (for example R32_SFLOAT -> R32_UINT): the snapshot is uploaded into a new
// image with the descriptor's exact Vulkan type, so no incompatible image view is created.
bool sampled_rtt_snapshot_byte_compatible(
    prosper::gpu::DataFormat format, uint32_t components,
    prosper::gpu::LiveTargetPixelFormat target_format);

// Reconstruct a packed R11G11B10 sampled surface from the renderer's canonical color snapshot.
// The renderer keeps float targets as RGBA16F and ordinary targets as RGBA8; compute descriptors
// can subsequently alias that same target as GFX10 10_11_11_FLOAT. This conversion restores the
// descriptor-visible texel representation without reading stale guest backing.
bool pack_live_target_r11g11b10(const prosper::gpu::LiveTargetSnapshot& snapshot,
                                uint8_t* packed, size_t packed_size);

// One reflected storage-buffer binding can occupy several Vulkan descriptors when the guest
// selects a V# from a runtime table. Keep the descriptor run as one shared plan so pool capacity,
// layout arity, and descriptor writes cannot derive the same count three different ways.
struct LiveComputeBufferBindingRun {
    size_t first_descriptor = 0;
    uint32_t descriptor_count = 0;
};

struct LiveComputeBufferDescriptorPlan {
    std::vector<LiveComputeBufferBindingRun> bindings;
    uint32_t total_descriptor_count = 0;
    bool valid = false;
};

// Validate and flatten the reflected/runtime storage-buffer arities. Descriptor arrays currently
// support read-only access; writable/atomic arrays reject until per-entry write authority is part of
// the public contract. Scalar bindings retain their historical one-descriptor plan.
LiveComputeBufferDescriptorPlan plan_live_compute_buffer_descriptors(
    const std::vector<prosper::gpu::SpirvDescriptorBinding>& descriptors,
    const prosper::gpu::ShaderResourceTable* resources,
    bool descriptor_indexing_support);

// Execute already-realized compute items synchronously. Exposed for the production-backend test.
bool execute_live_compute_items(const std::vector<prosper::gpu::ComputeItem>& items);

// Borrow an exact native storage result for a later sampled graphics binding. The import is
// deliberately narrower than the compute image cache: only a successful typed-storage dispatch can
// publish one, the complete descriptor identity must match, and either the current submit journal or
// the cache's page watch must prove that no later guest write overlapped the result. The lease pins
// the Vulkan image until the graphics submission has completed. Handles remain opaque here so this
// shared interface does not expose Vulkan types.
struct LiveComputeImageImport {
    uint32_t width = 0, height = 0, depth = 0;
    // Non-zero when `image` is a 2D-array result that a graphics cube descriptor must consume
    // through prosper's established vertical-face-stack lowering. The renderer copies each source
    // layer into one `width x (height * vertical_stack_layers)` sampled image on the GPU. Ordinary
    // direct imports keep this zero.
    uint32_t vertical_stack_layers = 0;
    // Global PM4 command ordinal of the storage dispatch that produced this image. A graphics
    // consumer can use it to merge a later renderer-owned attachment write without replacing the
    // other layers with stale guest memory.
    uint64_t producer_command_order = 0;
    uint32_t native_format = 0; // opaque VkFormat
    uint32_t layout = 0;        // opaque VkImageLayout; currently GENERAL
    void* image = nullptr;      // borrowed VkImage
    void* device = nullptr;     // VkDevice that owns image
    std::shared_ptr<void> lease;

    bool valid() const {
        return width && height && depth && native_format && image && device && lease;
    }
};
bool import_live_compute_storage_image(const prosper::gpu::ShaderResource& sampled_resource,
                                       uint64_t guest_bytes,
                                       LiveComputeImageImport& import);

// Exact Vulkan format a graphics sampled descriptor must request when attempting to lease a
// retained typed-storage result. Kept on the compute side so the publisher/importer and renderer
// cannot grow independent format allowlists. The integer is an opaque VkFormat just like
// LiveComputeImageImport::native_format; zero means the current graphics path must use guest bytes.
uint32_t live_compute_graphics_import_native_format(
    prosper::gpu::DataFormat format, uint32_t components);

// Exact guest-byte identity used to look up a retained storage image from a graphics sampled
// descriptor. Ordinary decoded textures already know their source span. A six-face R16 cube is
// different: the ordinary decoder reports one face, while the compute producer owns the complete
// 2D-array allocation. Returning the full descriptor footprint here keeps both cache keys equal.
uint64_t live_compute_graphics_import_guest_bytes(
    const prosper::gpu::ShaderResource& sampled_resource,
    uint64_t decoded_source_bytes);

// Monotonic diagnostic count of writable-buffer results whose exact GPU comparison avoided a host
// mapping/scan. Exposed so the production-backend test can prove that optimization path executes.
uint64_t live_compute_buffer_gpu_result_skips();
// #3155. Running totals of how cached compute sources were proven unchanged -- intra-submit GPU
// write journal, armed page watch, or a full byte compare -- and of what the promotion policy saw
// each time it was consulted. Always collected (per acquisition, not per byte);
// PROSPER_WATCH_PROMOTE_CENSUS additionally prints them every 256 submits and at exit. Read the
// `decisions=`/`validated=` denominators before any ratio: comparing two arms' totals taken at
// different points in a run is exactly the mistake that produced this issue's retracted numbers.
WriteWatchCensusSnapshot live_compute_write_watch_census();
// #3307. Running totals of the compute->graphics image borrow: why each graphics sampled
// descriptor did or did not lease the device image a compute dispatch had just produced, and -- on
// the producer side -- why an image was or was not published for borrowing at all. The two
// partitions together separate "the consumer's key finds nothing" from "nothing was ever
// published" from "the entry is published but no proof says guest memory still matches it", which
// is exactly the five-way branch a single `return false` used to collapse.
//
// Always collected. PROSPER_COMPUTE_BORROW_CENSUS additionally prints the totals every 256 submits
// and at exit, and enables the O(cache) near-miss scan that turns a lookup miss into a named key
// field. Read the `imports=` / `attempted=` / `publish_evaluated=` denominators before any ratio.
ComputeImageBorrowCensusSnapshot live_compute_image_borrow_census();
// The renderer's verdict on an import that succeeded: it re-checks format, extent and device, and
// a rejection there costs exactly what a miss costs. Recorded from `live_renderer.cpp` because only
// the consumer knows it happened.
void live_compute_record_image_borrow_renderer_verdict(bool accepted);
// Monotonic diagnostic count of retained sampled-image hits whose validated source omitted upload.
// Capture/replay tests use this to prove residency without relying on timing-sensitive assertions.
uint64_t live_compute_sampled_image_upload_skips();
// True only when an ordinary guest-backed 2D sampled image uses the same native Vulkan texel
// representation as its typed storage counterpart. This is the format half of the retained-image
// transfer contract; resource identity and write authority are checked separately at runtime.
bool compute_native_2d_transfer_format_compatible(prosper::gpu::DataFormat format,
                                                  uint32_t components);
// True only when an image resource declaring a compute mip chain can be materialized
// by the compute backend (excluding live renderer targets and unsupported dimensions).
bool compute_binding_mip_chain_materializable(const prosper::gpu::ShaderResource& r,
                                              bool renderer_owned);
struct ComputeSampledCacheSpan {
    uint64_t gpu_addr = 0;
    uint32_t guest_bytes = 0;
    bool eligible = false;
};
// Returns the validated allocation span and cache candidate eligibility for a sampled image resource,
// taking into account single-level and multi-level mip chains (#3291).
ComputeSampledCacheSpan compute_sampled_cache_span(
    const prosper::gpu::ShaderResource& r,
    uint32_t mip_levels,
    uint64_t sampled_guest_need,
    const prosper::gpu::MipChainPlan& mip_chain);
// Monotonic count of sampled 2D/3D images seeded from an exact retained native storage result with a
// device-local image copy instead of a guest-memory conversion/upload.
uint64_t live_compute_storage_transfer_seeds();
// Monotonic count of compressed storage results admitted only after an exact successful writeback
// published both ordinary base bytes and an all-uncompressed DCC metadata plane.
uint64_t live_compute_dcc_post_writeback_promotions();
// Monotonic witnesses for the compressed-producer allocation path. An exact unpinned entry can be
// forcibly reseeded in place; all other cases retain the post-writeback replacement fallback.
uint64_t live_compute_dcc_forced_seed_allocation_reuses();
uint64_t live_compute_dcc_post_writeback_replacements();
// Query the initialized live backend for the exact typed 3D storage+transfer image contract. Tests
// use this to exercise native-volume execution only on devices that can create the Vulkan image.
bool live_compute_native_storage_3d_supported(prosper::gpu::DataFormat format,
                                              uint32_t components,
                                              uint32_t width, uint32_t height,
                                              uint32_t depth);
// Monotonic count of exact compute fills executed by the structurally guarded CPU path. Tests use
// the delta to distinguish the intended bypass from a successful Vulkan fallback.
uint64_t live_compute_cpu_fill_dispatches();
// Monotonic attribution for storage-result source validation. Production-backend tests use this to
// prove that changing proven-full results do not copy a redundant guest-byte snapshot.
uint64_t live_compute_storage_result_snapshot_bytes();
// Monotonic attribution for the collision-free host fallback used when a retained storage result
// cannot keep an exact GPU comparison buffer. Eligible GPU baselines must not be copied here first.
uint64_t live_compute_image_result_snapshot_bytes();
// A cold, proven-full guest target has no observable old seed. Large targets may defer their exact
// source baseline until an address actually repeats; replay-owned and partial targets may not.
bool cold_storage_result_snapshot_can_defer(bool host_data, bool full_overwrite,
                                            size_t guest_bytes, size_t minimum_bytes);

// Deterministic failure injection for the storage-image recovery regression test. The next storage
// readback fails after dispatch, exercising retained-image invalidation without a Vulkan fault.
void live_compute_fail_next_storage_readback_for_test();

// Deterministic controls for post-DCC cache-promotion regressions. The first leaves the next
// writable metadata plane unresolved, exercising the final all-0xff recheck. The second disables
// one eligible allocation reuse so tests prove the replacement fallback actually runs. The third
// models a real replacement cache-capacity preflight after successful writeback.
void live_compute_leave_next_dcc_metadata_compressed_for_test();
void live_compute_disable_next_dcc_allocation_reuse_for_test();
void live_compute_limit_next_image_replacement_for_test();

// Deterministically lower the next eligible cold storage admission crossover to zero. This lets a
// compact production-backend fixture execute the real deferral predicate and retention branch.
void live_compute_zero_next_cold_storage_snapshot_minimum_for_test();

// Deterministic setup/ownership failure injection for the storage-result fallback regression. The
// pair models a transient compare-pipeline failure followed by a cache-capacity failure without
// relying on driver behavior.
void live_compute_force_next_image_result_host_fallback_for_test();
void live_compute_fail_next_image_result_buffer_retain_for_test();

// Deterministically model the permanent failure observed in Astro Bot without asking the driver to
// lose a real device. The attempt count proves both the current batch and later callbacks stop before
// another vkQueueSubmit after the injected failure.
void live_compute_force_next_queue_submit_device_lost_for_test();
uint64_t live_compute_queue_submit_attempts();

// Register the synchronous Vulkan compute backend used by AGC submit processing.
void register_live_compute();

// Publish the stable timing selector's final seen/matched verdict. Idempotent with the ordinary
// destructor fallback; prosper-app calls this before its deliberate _Exit teardown path.
void report_live_compute_timing_selector_summary();

// Persist the driver pipeline cache without tearing the compute context down. prosper-app cannot
// run C++/Vulkan destructors while its guest thread is detached, so its deliberate _Exit path calls
// this explicitly after requesting the guest stop.
void flush_live_compute_pipeline_cache();

} // namespace prosper::frontend
