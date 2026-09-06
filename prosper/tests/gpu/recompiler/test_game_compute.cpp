#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/resources/image_identity.hpp"
#include "gpu/resources/mip_chain_plan.hpp"
#include "gpu/texture/tile.hpp"
#include "host/memory/guest_write_watch.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/texture/seed_reprove.hpp"
#include "fixtures/gta5_cf9200_fixture.hpp"
#include "fixtures/test_scratch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#if defined(__linux__)
#include <csignal>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <sys/wait.h>
#endif

using namespace prosper::gpu;

static int fails = 0;
// #1690: this binary was excluded from CI for months, and the cost was argued from a *static* grep
// of CHECK sites because nothing reported how many actually ran. Count the executed ones so the
// coverage this registration contributes is a measured number in the run's own log, and so a gate
// that quietly stops executing assertions shows up as a drop rather than as unchanged green.
static int checks = 0;
#define CHECK(c, msg) do { ++checks; if (!(c)) { std::printf("FAIL: %s\n", msg); fails++; } } while (0)

#if defined(__linux__)
// ---------------------------------------------------------------------------------------------
// #3156: this suite arms real page-protection write watches, so it needs the one production
// signal behaviour that services them.
//
// `guest_write_watch_set_fault_onstack(true)` is an ASSERTION to the watch layer: "a SIGSEGV
// handler is installed, it runs on a sigaltstack, and it will service a write fault on a page I
// armed". This binary made that assertion in seven places while installing no handler at all, and
// compensated by calling `guest_write_watch_notify_host_write()` at the write sites it knew about.
// That works only while nothing arms a watch the test did not anticipate. It stopped working at
// `PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_HITS=1`, where the compute cache arms on a first
// acquisition: `large_result[0] ^= 0xffffffffu;` (the "externally changed buffer" step) then stored
// into a page the cache had just made read-only, and the process died -- `notl 0x0(%r13)` at a
// page-aligned mmap address, with `#0 main ()` as the whole backtrace. So the one setting that
// exercises promotion was the one setting the harness could not host, and nobody could measure
// promotion here.
//
// Deliberately NOT `prosper::install_trap_handler()`. That installs the guest-image fault
// machinery -- hardware breakpoints, the null-page companion mapping, lazy commit, guest-%fs
// scoping, siglongjmp recovery -- none of which this suite has any guest for. Model the single
// production behaviour that matters (exec_image_linux.cpp's write-watch branch) and keep every
// other fault fatal and loud, following `tests/host/test_dmem.cpp`'s precedent.
namespace {

// Distinct from the suite's own 1-on-assertion-failure so a fault can never be read as a FAIL, and
// from the production worker-fault _exit(90).
constexpr int kUnexpectedFaultExit = 86;

void write_watch_segv_handler(int sig, siginfo_t* info, void* context) {
#if defined(__x86_64__)
    // Production offers only WRITE faults to the watch (it tests the page-fault error code's bit
    // 1). An armed page keeps PROT_READ, so a read fault is never a watch event; leaving it fatal
    // here keeps the handler from resuming a fault it cannot actually have caused.
    const bool write_fault =
        context != nullptr &&
        (static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_ERR] & 2) != 0;
#else
    const bool write_fault = context != nullptr;
#endif
    if (sig == SIGSEGV && write_fault && info != nullptr && info->si_addr != nullptr &&
        prosper::host::guest_write_watch_handle_fault(
            reinterpret_cast<uint64_t>(info->si_addr)))
        return;
    // Not a page we armed. Returning here would re-execute the faulting instruction forever and
    // turn a real defect into a hang; a handler that swallows is worse than the crash it replaced.
    static constexpr char message[] = "test_game_compute: unexpected SIGSEGV (not an armed "
                                      "write-watch page)\n";
    (void)!write(STDERR_FILENO, message, sizeof(message) - 1);
    _exit(kUnexpectedFaultExit);
}

bool install_write_watch_segv_handler() {
    // The watch layer only arms when the handler is on an alternate stack: it resolves a fault by
    // returning, so without SA_ONSTACK the kernel writes the signal frame into the faulting store's
    // SysV red zone.
    //
    // Main thread only, which is sufficient HERE and would not be in production. `sigaltstack` is
    // per-thread; production installs one per guest worker from its trampoline. This binary has no
    // guest, and the only writes into watched guest pages come from the test's own thread or from
    // backend paths that call `guest_write_watch_notify_host_write` first. The parallel copy/compare
    // workers touch mapped Vulkan memory, never an armed guest page. If that ever stops being true,
    // this needs the per-thread install too.
    static uint8_t alternate_stack[256 * 1024];
    stack_t stack{};
    stack.ss_sp = alternate_stack;
    stack.ss_size = sizeof(alternate_stack);
    struct sigaction action{};
    action.sa_sigaction = write_watch_segv_handler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    return sigaltstack(&stack, nullptr) == 0 && sigaction(SIGSEGV, &action, nullptr) == 0;
}

// Arm 2 of the harness fix. Proves the handler is scoped rather than absorbing: a write fault on a
// page nothing armed must still end the process loudly. Run in a child so the proof is executable
// rather than a comment. `alarm` bounds the "handler wrongly resumed" case, which would otherwise
// re-execute the store forever and hang ctest instead of failing it.
int unarmed_fault_child_status() {
    std::fflush(stdout);
    std::fflush(stderr);
    const pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        alarm(20);
        void* mapping = mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) _exit(70);
        *static_cast<volatile uint8_t*>(mapping) = 0x11;
        _exit(0);   // reached only if the handler swallowed the fault
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return -1;
    return status;
}

} // namespace
#endif  // __linux__

static int file_descriptor(FILE* file) {
#ifdef _WIN32
    return _fileno(file);
#else
    return fileno(file);
#endif
}

static int duplicate_descriptor(int fd) {
#ifdef _WIN32
    return _dup(fd);
#else
    return dup(fd);
#endif
}

static int replace_descriptor(int from, int to) {
#ifdef _WIN32
    return _dup2(from, to);
#else
    return dup2(from, to);
#endif
}

static void close_descriptor(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif
}

int main() {
#ifdef _WIN32
    _putenv_s("PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB", "0");
    _putenv_s("PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB", "1");
    _putenv_s("PROSPER_NO_DISK_PIPELINE_CACHE", "1");
    _putenv_s("PROSPER_COMPUTE_BORROW_CENSUS", "1");
#else
    setenv("PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB", "0", 1);
    setenv("PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB", "1", 1);
    setenv("PROSPER_NO_DISK_PIPELINE_CACHE", "1", 1);
    // #3307. The borrow census counts unconditionally, but its near-miss key SCAN is O(cache) and
    // therefore gated on the same variable as the report. The typed-storage borrow arms below
    // assert that scan names the right field, so it has to be armed before the first import.
    setenv("PROSPER_COMPUTE_BORROW_CENSUS", "1", 1);
#endif

#if defined(__linux__)
    // #3156. Install before anything can arm a watch -- every `set_fault_onstack(true)` below is a
    // promise that this handler exists, and the compute cache arms on its own schedule, not the
    // test's.
    CHECK(install_write_watch_segv_handler(),
          "write-watch SIGSEGV handler installed on an alternate stack");

    // Arm 1: the fault path is serviced in THIS binary. A raw architectural store into an armed
    // page, with no `guest_write_watch_notify_host_write` hook to disarm it first -- exactly the
    // shape that killed the process at PROMOTE_HITS=1. The store must land, the watch must read
    // Dirty, and the run must continue.
    {
        constexpr size_t fault_probe_bytes = 4096;
        auto* fault_probe = static_cast<uint8_t*>(mmap(
            nullptr, fault_probe_bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        CHECK(fault_probe != MAP_FAILED, "map write-watch fault-probe range");
        if (fault_probe != MAP_FAILED) {
            std::memset(fault_probe, 0x24, fault_probe_bytes);
            prosper::host::guest_write_watch_set_fault_onstack(true);
            prosper::host::guest_write_watch_notify_direct_mapping_added(
                reinterpret_cast<uint64_t>(fault_probe), fault_probe_bytes, 0x7f0000,
                0x3 /* SCE CPU_READ|CPU_WRITE */);
            auto probe_watch = prosper::host::GuestWriteWatch::create(
                reinterpret_cast<uint64_t>(fault_probe), fault_probe_bytes);
            CHECK(static_cast<bool>(probe_watch) &&
                      probe_watch.query() == prosper::host::GuestWriteWatchQuery::Unchanged,
                  "fault probe arms a page-protection watch that starts Unchanged");
            *static_cast<volatile uint8_t*>(fault_probe + 64) = 0x5a;
            CHECK(fault_probe[64] == 0x5a,
                  "an unhooked CPU store into an armed page is serviced and lands");
            CHECK(probe_watch.query() == prosper::host::GuestWriteWatchQuery::Dirty,
                  "the serviced fault marks the armed watch Dirty");
            probe_watch.reset();
            prosper::host::guest_write_watch_notify_direct_mapping_removed(
                reinterpret_cast<uint64_t>(fault_probe), fault_probe_bytes);
            prosper::host::guest_write_watch_set_fault_onstack(false);
            munmap(fault_probe, fault_probe_bytes);
        }

        // Arm 2: the handler is scoped, not absorbing. A write fault on a page nothing armed must
        // still kill the process loudly, or this fix would trade a deterministic crash for silently
        // green runs over corrupt memory.
        const int unarmed_status = unarmed_fault_child_status();
        CHECK(unarmed_status >= 0 && WIFEXITED(unarmed_status) &&
                  WEXITSTATUS(unarmed_status) == kUnexpectedFaultExit,
              "a write fault on an unarmed page remains fatal and reported");
    }
#endif
    const bool adaptive_storage_result_validation_enabled =
        std::getenv("PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION") == nullptr;
    const bool native_2d_compute_transfer_enabled =
        std::getenv("PROSPER_NO_NATIVE_2D_COMPUTE_TRANSFER") == nullptr;
    const bool native_2d_compute_transfer_available =
        native_2d_compute_transfer_enabled &&
        adaptive_storage_result_validation_enabled;
    const bool cold_storage_snapshot_deferral_enabled =
        adaptive_storage_result_validation_enabled;
    using prosper::frontend::ComputeImageCacheClass;
    CHECK(prosper::frontend::compute_image_cache_default_minimum_bytes(
              ComputeImageCacheClass::sampled) == 4ull * 1024ull &&
          prosper::frontend::compute_image_cache_default_minimum_bytes(
              ComputeImageCacheClass::storage) == 4ull * 1024ull,
          "sampled and storage images share 4 KiB host-page default cache crossover");
    CHECK(!prosper::frontend::compute_image_cache_default_eligible(
              4ull * 1024ull - 1, ComputeImageCacheClass::sampled) &&
          prosper::frontend::compute_image_cache_default_eligible(
              4ull * 1024ull, ComputeImageCacheClass::sampled) &&
          !prosper::frontend::compute_image_cache_default_eligible(
              4ull * 1024ull - 1, ComputeImageCacheClass::storage) &&
          prosper::frontend::compute_image_cache_default_eligible(
              4ull * 1024ull, ComputeImageCacheClass::storage),
          "image residency policy includes each crossover exactly without caching smaller inputs");
    CHECK(prosper::frontend::compute_image_cache_default_limit_bytes(0) ==
              512ull * 1024ull * 1024ull &&
          prosper::frontend::compute_image_cache_default_limit_bytes(
              4ull * 1024ull * 1024ull * 1024ull) == 512ull * 1024ull * 1024ull &&
          prosper::frontend::compute_image_cache_default_limit_bytes(
              8ull * 1024ull * 1024ull * 1024ull) == 1024ull * 1024ull * 1024ull &&
          prosper::frontend::compute_image_cache_default_limit_bytes(
              16ull * 1024ull * 1024ull * 1024ull) == 2048ull * 1024ull * 1024ull &&
          prosper::frontend::compute_image_cache_default_limit_bytes(UINT64_MAX) ==
              2048ull * 1024ull * 1024ull,
          "image cache limit scales with local memory and retains bounded floor and ceiling");
    CHECK(!prosper::frontend::compute_pipeline_is_large(32u * 1024u - 1u) &&
          prosper::frontend::compute_pipeline_is_large(32u * 1024u),
          "large portable compute modules retain an exact diagnostic policy boundary");
    std::array<uint8_t, 32> pipeline_cache_header{};
    const uint32_t pipeline_header_size = static_cast<uint32_t>(pipeline_cache_header.size());
    const uint32_t pipeline_header_version = 1;
    const uint32_t pipeline_vendor = 0x10de;
    const uint32_t pipeline_device = 0x2b85;
    std::array<uint8_t, 16> pipeline_uuid{};
    for (size_t i = 0; i < pipeline_uuid.size(); ++i)
        pipeline_uuid[i] = static_cast<uint8_t>(i * 7u + 3u);
    std::memcpy(pipeline_cache_header.data() + 0, &pipeline_header_size, sizeof(uint32_t));
    std::memcpy(pipeline_cache_header.data() + 4, &pipeline_header_version, sizeof(uint32_t));
    std::memcpy(pipeline_cache_header.data() + 8, &pipeline_vendor, sizeof(uint32_t));
    std::memcpy(pipeline_cache_header.data() + 12, &pipeline_device, sizeof(uint32_t));
    std::memcpy(pipeline_cache_header.data() + 16, pipeline_uuid.data(), pipeline_uuid.size());
    CHECK(prosper::frontend::compute_pipeline_cache_blob_compatible(
              pipeline_cache_header.data(), pipeline_cache_header.size(),
              pipeline_vendor, pipeline_device, pipeline_uuid.data(), pipeline_uuid.size()),
          "disk compute pipeline cache accepts its exact Vulkan device identity");
    pipeline_cache_header[16] ^= 1u;
    CHECK(!prosper::frontend::compute_pipeline_cache_blob_compatible(
              pipeline_cache_header.data(), pipeline_cache_header.size(),
              pipeline_vendor, pipeline_device, pipeline_uuid.data(), pipeline_uuid.size()),
          "disk compute pipeline cache rejects a mismatched driver UUID");
    pipeline_cache_header[16] ^= 1u;
    CHECK(!prosper::frontend::compute_pipeline_cache_blob_compatible(
              pipeline_cache_header.data(), pipeline_cache_header.size() - 1,
              pipeline_vendor, pipeline_device, pipeline_uuid.data(), pipeline_uuid.size()),
          "disk compute pipeline cache rejects a truncated Vulkan header");
    CHECK(prosper::frontend::compute_sampled_guest_prepare_required(false, false, false),
          "guest-backed sampled image prepares its source");
    CHECK(!prosper::frontend::compute_sampled_guest_prepare_required(false, true, false),
          "renderer-owned sampled image uses renderer authority instead of guest backing");
    CHECK(!prosper::frontend::compute_sampled_guest_prepare_required(false, false, true),
          "directly imported depth image does not validate unused guest backing");
    CHECK(prosper::frontend::compute_sampled_guest_prepare_required(
              false, false, true, true),
          "imported-image recovery switch restores guest validation");
    CHECK(!prosper::frontend::compute_sampled_guest_prepare_required(true, false, false),
          "storage image retains its independent seed/writeback path");

    // #3407: the renderer conversion cache. `compute_sampled_guest_prepare_required` above is why
    // this exists -- it excludes renderer-owned surfaces from guest validation, correctly, and the
    // consequence was that they had no cache identity at all and were reconverted every dispatch.
    //
    // Every arm below is a MUTATION of one admitting case, so each clause is shown to be
    // load-bearing rather than merely present. Without that, a predicate that returned `true` for
    // everything would pass an "it admits the good case" test.
    {
        using prosper::frontend::ComputeRendererConversionCacheInputs;
        using prosper::frontend::compute_renderer_conversion_cache_candidate;
        const ComputeRendererConversionCacheInputs admit{
            .renderer_owned = true,
            .has_published_pixels = true,
            .cache_enabled = true,
            .persistent_enabled = true,
        };
        CHECK(compute_renderer_conversion_cache_candidate(admit),
              "a 2D renderer-owned sampled publication is a conversion-cache candidate");

        const auto refuses = [&](auto mutate) {
            ComputeRendererConversionCacheInputs in = admit;
            mutate(in);
            return !compute_renderer_conversion_cache_candidate(in);
        };
        CHECK(refuses([](auto& in) { in.renderer_owned = false; }),
              "a guest-backed surface is not a conversion-cache candidate");
        CHECK(refuses([](auto& in) { in.has_published_pixels = false; }),
              "a renderer surface with no published buffer has no cache identity");
        CHECK(refuses([](auto& in) { in.storage_image = true; }),
              "a storage image keeps its own writeback-driven cache path");
        CHECK(refuses([](auto& in) { in.imported = true; }),
              "a directly imported image is the renderer's, not ours to cache");
        CHECK(refuses([](auto& in) { in.depth_bits_source = true; }),
              "a borrowed depth plane is not converted from a publication");
        CHECK(refuses([](auto& in) { in.seed_skip = true; }),
              "a write-only target is never seeded, so there is nothing to cache");
        CHECK(refuses([](auto& in) { in.compute_transfer_seed_borrowed = true; }),
              "a compute-transfer seed already supplies the image device-side");
        CHECK(refuses([](auto& in) { in.seeded_from_imported = true; }),
              "an image seeded by copy from the renderer needs no host conversion");
        CHECK(refuses([](auto& in) { in.volume_or_array = true; }),
              "volume and array shapes are excluded from the conversion cache");
        CHECK(refuses([](auto& in) { in.array_layers = 2; }),
              "a multi-layer publication is excluded from the conversion cache");
        CHECK(refuses([](auto& in) { in.mip_levels = 2; }),
              "a mip chain is excluded from the conversion cache");
        CHECK(refuses([](auto& in) { in.cache_enabled = false; }),
              "PROSPER_NO_RENDERER_CONVERSION_CACHE disables the cache");
        CHECK(refuses([](auto& in) { in.persistent_enabled = false; }),
              "a surface below the persistent-image threshold is not cached");
    }

    // The cache's entire correctness argument: a hit requires the SAME published buffer, and the
    // entry must already hold a completed conversion of it. The identity is a pointer only because
    // the cache holds the shared_ptr -- which is what makes the buffer neither freeable (no address
    // reuse, no ABA) nor writable (const). A wrong `true` here binds a stale image, so each clause
    // gets its own counter-arm.
    {
        using prosper::frontend::compute_renderer_conversion_cache_hit;
        const std::vector<uint8_t> publication_a(4, 0u);
        const std::vector<uint8_t> publication_b(4, 0u);
        const void* const a = &publication_a;
        const void* const b = &publication_b;
        CHECK(compute_renderer_conversion_cache_hit(true, a, a),
              "the same publication on a validated entry authorizes the upload skip");
        CHECK(!compute_renderer_conversion_cache_hit(true, a, b),
              "a different publication must reconvert, even at the same address and size");
        CHECK(!compute_renderer_conversion_cache_hit(false, a, a),
              "an entry whose transfer has not completed cannot authorize a skip");
        CHECK(!compute_renderer_conversion_cache_hit(true, nullptr, a),
              "an entry holding no publication cannot authorize a skip");
        CHECK(!compute_renderer_conversion_cache_hit(true, a, nullptr),
              "a binding with no publication cannot take a skip");
        CHECK(!compute_renderer_conversion_cache_hit(true, nullptr, nullptr),
              "two absent publications are not a match");
    }
    CHECK(prosper::frontend::storage_writeback_can_tile_mapped_bytes(
              true, 27, false, false),
          "exact-width tiled storage can feed mapped bytes directly to the tiler");
    CHECK(!prosper::frontend::storage_writeback_can_tile_mapped_bytes(
              false, 27, false, false),
          "converted storage retains its mutable packed buffer");
    CHECK(!prosper::frontend::storage_writeback_can_tile_mapped_bytes(
              true, 0, false, false),
          "linear guest storage still copies mapped bytes into guest memory");
    CHECK(!prosper::frontend::storage_writeback_can_tile_mapped_bytes(
              true, 27, true, false),
          "poison proving retains a mutable copy for untouched-texel restoration");
    CHECK(!prosper::frontend::storage_writeback_can_tile_mapped_bytes(
              true, 27, false, true),
          "the recovery switch restores the copied writeback path");

    bool half_luts_match = true;
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        const uint16_t half = static_cast<uint16_t>(bits);
        const float value = half_to_float(half);
        uint32_t float_bits = 0;
        std::memcpy(&float_bits, &value, sizeof(float_bits));
        half_luts_match &= prosper::frontend::storage_unpack_float16_bits(half) == float_bits;

        float normalized = value;
        if (std::isnan(normalized)) normalized = 0.0f;
        normalized = normalized < 0.0f ? 0.0f : (normalized > 1.0f ? 1.0f : normalized);
        const uint8_t reference = static_cast<uint8_t>(std::lround(normalized * 255.0f));
        half_luts_match &= prosper::frontend::sampled_float16_to_unorm8(half) == reference;
    }
    CHECK(half_luts_match,
          "binary16 lookup conversions are exhaustive bit-exact matches for storage and sampling");

    std::vector<uint8_t> half_source(65536u * sizeof(uint16_t));
    std::vector<uint8_t> half_rgba(65536u * 4u);
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        const uint16_t half = static_cast<uint16_t>(bits);
        std::memcpy(half_source.data() + bits * sizeof(half), &half, sizeof(half));
    }
    prosper::frontend::sampled_float16_to_unorm8_range(
        half_source.data(), 1, 65536u, half_rgba.data());
    bool half_range_matches = true;
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        half_range_matches &=
            half_rgba[bits * 4] == prosper::frontend::sampled_float16_to_unorm8(
                                       static_cast<uint16_t>(bits)) &&
            half_rgba[bits * 4 + 1] == 0 && half_rgba[bits * 4 + 2] == 0 &&
            half_rgba[bits * 4 + 3] == 255;
    }
    CHECK(half_range_matches,
          "parallel binary16 sampled range matches every scalar value and fills (R,0,0,1)");

    // Exercise the packed RGBA fast path with every binary16 bit pattern. On x86 this runtime-
    // dispatches through F16C/AVX2 when available; elsewhere it proves the identical scalar fallback.
    std::vector<uint8_t> half_source_x4(65536u * sizeof(uint16_t));
    std::vector<uint8_t> half_rgba_x4(65536u);
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        const uint16_t half = static_cast<uint16_t>(bits);
        std::memcpy(half_source_x4.data() + bits * sizeof(half), &half, sizeof(half));
    }
    prosper::frontend::sampled_float16_to_unorm8_range(
        half_source_x4.data(), 4, 65536u / 4u, half_rgba_x4.data());
    bool half_range_x4_matches = true;
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits)
        half_range_x4_matches &= half_rgba_x4[bits] ==
            prosper::frontend::sampled_float16_to_unorm8(static_cast<uint16_t>(bits));
    CHECK(half_range_x4_matches,
          "packed RGBA binary16 sampled range matches every scalar lookup value");

    // Compute's ordinary guest-backed 2D RGBA16F sampled path uses an RGBA8 staging image. A
    // uniform embedded DCC clear is authoritative over the compressed base allocation, so prove
    // that the narrow fast-clear gate produces exactly the same sampled bytes as an explicitly
    // materialized uncompressed FP16 image. DCC_CLEAR_0001 is especially important: its alpha-one
    // value is absent from a zero-filled compressed base and is consumed by Plucky's composite.
    {
        constexpr uint32_t clear_width = 64;
        constexpr size_t clear_texels = clear_width;
        ShaderResource clear_resource{};
        clear_resource.cls = ResourceClass::Texture;
        clear_resource.format = DataFormat::Float16;
        clear_resource.num_components = 4;
        clear_resource.img_dim = 1;
        clear_resource.width = clear_width;
        clear_resource.height = clear_resource.depth = 1;
        clear_resource.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
        clear_resource.size = static_cast<uint32_t>(
            tiled_surface_bytes(clear_width, 1, clear_resource.tile_mode, 0, 8));
        clear_resource.compression_enabled = true;
        clear_resource.meta_pipe_aligned = true;
        clear_resource.alpha_is_on_msb = true;
        clear_resource.metadata_addr = 0x301758d000ull;
        const size_t clear_metadata_bytes = gpu_capture_dcc_metadata_footprint(clear_resource);
        std::vector<uint8_t> clear_metadata(clear_metadata_bytes, 0x40);
        std::vector<uint8_t> clear_rgba(clear_texels * 4, 0xa5);
        uint8_t clear_code = 0;
        CHECK(clear_metadata_bytes &&
              prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, false, false, clear_rgba.data(), clear_texels,
                  clear_metadata.data(), clear_metadata.size(), &clear_code) &&
              clear_code == 0x40,
              "compute recognizes a complete uniform RGBA16F DCC_CLEAR_0001 plane");

        std::vector<uint8_t> explicit_fp16(clear_texels * 8, 0);
        const uint16_t half_one = float_to_half(1.0f);
        for (size_t texel = 0; texel < clear_texels; ++texel)
            std::memcpy(explicit_fp16.data() + texel * 8 + 6,
                        &half_one, sizeof(half_one));
        std::vector<uint8_t> explicit_rgba(clear_texels * 4, 0);
        prosper::frontend::sampled_float16_to_unorm8_range(
            explicit_fp16.data(), 4, clear_texels, explicit_rgba.data());
        CHECK(clear_rgba == explicit_rgba,
              "DCC_CLEAR_0001 matches explicitly materialized FP16 (0,0,0,1)");
        bool clear_channels_match = true;
        for (size_t texel = 0; texel < clear_texels; ++texel)
            clear_channels_match &= clear_rgba[texel * 4 + 0] == 0 &&
                                    clear_rgba[texel * 4 + 1] == 0 &&
                                    clear_rgba[texel * 4 + 2] == 0 &&
                                    clear_rgba[texel * 4 + 3] == 255;
        CHECK(clear_channels_match,
              "MSB alpha placement preserves the DCC alpha-one sampled channel");

        ShaderResource lsb_alpha = clear_resource;
        lsb_alpha.alpha_is_on_msb = false;
        uint8_t lsb_pixel[4]{};
        CHECK(prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  lsb_alpha, true, false, false, lsb_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()) &&
              std::vector<uint8_t>(lsb_pixel, lsb_pixel + 4) ==
                  std::vector<uint8_t>({255, 0, 0, 0}),
              "LSB alpha placement routes DCC_CLEAR_0001 to component zero");

        std::vector<uint8_t> rejected = clear_metadata;
        rejected.back() = 0x00;
        uint8_t rejected_pixel[4]{};
        CHECK(!prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, false, false, rejected_pixel, 1,
                  rejected.data(), rejected.size()) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, false, false, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size() - 1) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, false, false, false, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, true, false, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, false, true, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()),
              "mixed, incomplete, non-guest, arrayed, and rollback DCC states retain the old path");
        std::fill(rejected.begin(), rejected.end(), 0xff);
        ShaderResource wrong_shape = clear_resource;
        wrong_shape.declared_mip_levels = 2;
        ShaderResource wrong_format = clear_resource;
        wrong_format.format = DataFormat::Unorm8;
        CHECK(!prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, false, false, rejected_pixel, 1,
                  rejected.data(), rejected.size()) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  wrong_shape, true, false, false, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  wrong_format, true, false, false, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()),
              "uncompressed metadata, mip chains, and non-FP16 views fail closed");
    }

    std::vector<uint32_t> half_storage_x4(65536u);
    prosper::frontend::storage_unpack_float16x4_range(
        half_source_x4.data(), 65536u / 4u, half_storage_x4.data());
    bool half_storage_x4_matches = true;
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits)
        half_storage_x4_matches &= half_storage_x4[bits] ==
            prosper::frontend::storage_unpack_float16_bits(static_cast<uint16_t>(bits));
    CHECK(half_storage_x4_matches,
          "packed RGBA16F storage range preserves every scalar float bit pattern");

    // The storage writeback fast path converts two RGBA32F texels per F16C instruction. Compare a
    // million deterministic float bit patterns, including explicit overflow/subnormal/NaN payload
    // edges, against the established scalar round-to-nearest-even converter.
    constexpr size_t pack_channels_count = 1u << 20;
    std::vector<uint32_t> pack_channels(pack_channels_count);
    const uint32_t pack_edges[] = {
        0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
        0x33000000u, 0x33000001u, 0x387fc000u, 0x38800000u,
        0x477fe000u, 0x477ff000u, 0x47800000u, 0xc7800000u,
        0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u,
        0x7fffffffu, 0xff800001u, 0x3f800000u, 0xbf800000u,
    };
    size_t pack_at = 0;
    for (uint32_t bits : pack_edges) pack_channels[pack_at++] = bits;
    uint32_t pack_random = 0x91e10da5u;
    while (pack_at < pack_channels.size()) {
        pack_random ^= pack_random << 13;
        pack_random ^= pack_random >> 17;
        pack_random ^= pack_random << 5;
        pack_channels[pack_at++] = pack_random;
    }
    std::vector<uint8_t> packed_half(pack_channels_count * sizeof(uint16_t));
    prosper::frontend::storage_pack_float16x4_range(
        pack_channels.data(), pack_channels_count / 4, packed_half.data());
    bool storage_half_range_matches = true;
    for (size_t i = 0; i < pack_channels.size(); ++i) {
        float value;
        std::memcpy(&value, &pack_channels[i], sizeof(value));
        const uint16_t expected = prosper::gpu::float_to_half(value);
        uint16_t actual = 0;
        std::memcpy(&actual, packed_half.data() + i * sizeof(actual), sizeof(actual));
        if (actual != expected) {
            std::printf("Float16 pack mismatch index=%zu f32=%08x expected=%04x actual=%04x\n",
                        i, pack_channels[i], expected, actual);
            storage_half_range_matches = false;
            break;
        }
    }
    CHECK(storage_half_range_matches,
          "packed RGBA32F storage range matches scalar Float16 rounding and NaN payloads");

    // pack_float32_to_rgba16f_range replaced a per-texel/per-channel scalar loop in the renderer's
    // sampled-Float32 path. The contract it has to keep is not "close enough": it is the EXACT
    // output of that loop, including the (0,0,0,1) synthesis for channels the source does not carry,
    // for every component count and for a padded source stride. The reference below is that original
    // loop, written out here rather than described, so the assertion compares implementations rather
    // than comparing the new code against a restatement of what it was meant to do.
    //
    // The value population deliberately reuses `pack_channels` above -- a million patterns seeded
    // with explicit NaN-payload, signaling-NaN, infinity, subnormal, negative-zero and
    // rounding-boundary edges. A fresh random population would be a weaker input, not a stronger
    // one, because those edges are exactly where a vector converter and a scalar one diverge.
    {
        auto scalar_reference = [](const uint8_t* source, uint32_t components,
                                   size_t source_texel_bytes, size_t texels,
                                   std::vector<uint8_t>& out) {
            out.assign(texels * 8, 0u);
            for (size_t t = 0; t < texels; ++t)
                for (uint32_t c = 0; c < 4; ++c) {
                    float value = c == 3 ? 1.0f : 0.0f;
                    if (c < components)
                        std::memcpy(&value, source + t * source_texel_bytes + c * 4, 4);
                    const uint16_t half = prosper::gpu::float_to_half(value);
                    std::memcpy(out.data() + t * 8 + c * 2, &half, 2);
                }
        };

        bool narrowing_matches = true;
        size_t narrowing_cases = 0;
        // components/stride pairs: the three the renderer actually produces (bpt = nc * 4 for
        // nc = 1, 2, 4), the awkward three-channel case, and a PADDED stride where the source texel
        // is wider than the channels it carries -- the one shape a "stride == components * 4"
        // assumption would silently read wrong.
        const struct { uint32_t components; size_t stride; } narrowing_shapes[] = {
            {1u, 4u}, {2u, 8u}, {3u, 12u}, {4u, 16u}, {2u, 16u}, {1u, 16u},
        };
        for (const auto& shape : narrowing_shapes) {
            const size_t texels = pack_channels_count / 4u;   // enough source words for stride 16
            const auto* const source = reinterpret_cast<const uint8_t*>(pack_channels.data());
            std::vector<uint8_t> expected;
            scalar_reference(source, shape.components, shape.stride, texels, expected);
            std::vector<uint8_t> actual(texels * 8, 0xccu);
            prosper::frontend::pack_float32_to_rgba16f_range(
                source, shape.components, shape.stride, texels, actual.data());
            ++narrowing_cases;
            if (actual != expected) {
                size_t first = 0;
                while (first < expected.size() && expected[first] == actual[first]) ++first;
                std::printf("Float32->RGBA16F narrowing mismatch components=%u stride=%zu "
                            "byte=%zu expected=%02x actual=%02x\n",
                            shape.components, shape.stride, first,
                            first < expected.size() ? expected[first] : 0u,
                            first < actual.size() ? actual[first] : 0u);
                narrowing_matches = false;
                break;
            }
        }
        CHECK(narrowing_matches && narrowing_cases == 6,
              "Float32->RGBA16F narrowing is byte-identical to the scalar loop it replaced, for "
              "every component count and for a padded source stride");

        // Refusals, so a malformed call cannot half-fill a destination: the guard has to reject
        // BEFORE writing anything, which a caller can only observe by the buffer staying untouched.
        std::vector<uint8_t> untouched(64u, 0x5au);
        const std::vector<uint8_t> untouched_before = untouched;
        const auto* const source = reinterpret_cast<const uint8_t*>(pack_channels.data());
        prosper::frontend::pack_float32_to_rgba16f_range(source, 0u, 4u, 8u, untouched.data());
        prosper::frontend::pack_float32_to_rgba16f_range(source, 2u, 0u, 8u, untouched.data());
        // stride narrower than the channels it claims to carry: reading it would run off the texel
        prosper::frontend::pack_float32_to_rgba16f_range(source, 4u, 8u, 8u, untouched.data());
        prosper::frontend::pack_float32_to_rgba16f_range(nullptr, 4u, 16u, 8u, untouched.data());
        prosper::frontend::pack_float32_to_rgba16f_range(source, 4u, 16u, 0u, untouched.data());
        CHECK(untouched == untouched_before,
              "Float32->RGBA16F narrowing writes nothing for a zero, null, or short-stride call");
    }

    using prosper::frontend::direct_sampled_rtt_compatible;
    CHECK(direct_sampled_rtt_compatible(DataFormat::Unorm8, 4,
                                        LiveTargetPixelFormat::Rgba8Unorm, false) &&
          direct_sampled_rtt_compatible(DataFormat::Float16, 4,
                                        LiveTargetPixelFormat::Rgba16Float, false) &&
          direct_sampled_rtt_compatible(DataFormat::Float16, 2,
                                        LiveTargetPixelFormat::Rg16Float, false) &&
          direct_sampled_rtt_compatible(DataFormat::Float16, 1,
                                        LiveTargetPixelFormat::R16Float, false) &&
          direct_sampled_rtt_compatible(DataFormat::Float10_11_11, 3,
                                        LiveTargetPixelFormat::R11G11B10Float, false),
          "renderer RTT direct bind accepts exact RGBA8, RG16F, RGBA16F, and R11G11B10 views");
    CHECK(!direct_sampled_rtt_compatible(DataFormat::Float16, 4,
                                         LiveTargetPixelFormat::Rgba8Unorm, true) &&
          !direct_sampled_rtt_compatible(DataFormat::Unorm8, 4,
                                         LiveTargetPixelFormat::Rgba16Float, true) &&
          !direct_sampled_rtt_compatible(DataFormat::Float16, 2,
                                         LiveTargetPixelFormat::Rgba16Float, true),
          "renderer RTT direct bind rejects format conversion and component aliases");
    CHECK(direct_sampled_rtt_compatible(DataFormat::Unorm16, 4,
                                        LiveTargetPixelFormat::Rgba8Unorm, true) &&
          !direct_sampled_rtt_compatible(DataFormat::Unorm16, 4,
                                         LiveTargetPixelFormat::Rgba8Unorm, false) &&
          !direct_sampled_rtt_compatible(DataFormat::Unorm16, 2,
                                         LiveTargetPixelFormat::Rgba8Unorm, true),
          "float RGBA16-UNORM sampled values may reuse RGBA8 without widening texels");
    CHECK(direct_sampled_rtt_compatible(DataFormat::Unorm8, 1,
                                        LiveTargetPixelFormat::R8Unorm, true) &&
          direct_sampled_rtt_compatible(DataFormat::Unorm8, 2,
                                        LiveTargetPixelFormat::Rg8Unorm, true) &&
          direct_sampled_rtt_compatible(DataFormat::Uint32, 1,
                                        LiveTargetPixelFormat::R32Uint, false) &&
          direct_sampled_rtt_compatible(DataFormat::Float32, 1,
                                        LiveTargetPixelFormat::R32Float, true) &&
          !direct_sampled_rtt_compatible(DataFormat::Float32, 1,
                                         LiveTargetPixelFormat::R32Uint, true),
          "single-channel live RTT imports require an exact Vulkan sampled type");
    using prosper::frontend::sampled_rtt_snapshot_byte_compatible;
    CHECK(sampled_rtt_snapshot_byte_compatible(
              DataFormat::Uint32, 1, LiveTargetPixelFormat::R32Float) &&
          sampled_rtt_snapshot_byte_compatible(
              DataFormat::Float32, 1, LiveTargetPixelFormat::R32Uint) &&
          sampled_rtt_snapshot_byte_compatible(
              DataFormat::Uint32, 4, LiveTargetPixelFormat::Rgba32Float) &&
          sampled_rtt_snapshot_byte_compatible(
              DataFormat::Float16, 2, LiveTargetPixelFormat::Rg16Float) &&
          sampled_rtt_snapshot_byte_compatible(
              DataFormat::Float16, 1, LiveTargetPixelFormat::R16Float) &&
          !sampled_rtt_snapshot_byte_compatible(
              DataFormat::Uint32, 1, LiveTargetPixelFormat::Rgba32Float) &&
          !sampled_rtt_snapshot_byte_compatible(
              DataFormat::Float16, 2, LiveTargetPixelFormat::R32Float),
           "renderer RTT snapshots preserve exact-width float/integer bit aliases only");
    // A renderer-owned target is stored canonically as RGBA8 or RGBA16F, while a later compute
    // descriptor can alias it as packed R11G11B10. Reconstruct the descriptor-visible words rather
    // than sampling stale guest backing or dropping the dispatch.
    {
        auto rgba8 = std::make_shared<std::vector<uint8_t>>(std::initializer_list<uint8_t>{
            255, 0, 128, 17, 0, 255, 64, 99,
        });
        LiveTargetSnapshot snapshot8{2, 1, LiveTargetPixelFormat::Rgba8Unorm, rgba8};
        std::vector<uint8_t> packed8(8);
        CHECK(prosper::frontend::pack_live_target_r11g11b10(
                  snapshot8, packed8.data(), packed8.size()),
              "RGBA8 renderer target reconstructs as two packed R11G11B10 texels");
        uint32_t words8[2]{};
        if (packed8.size() == sizeof(words8)) std::memcpy(words8, packed8.data(), sizeof(words8));
        CHECK((words8[0] & 0x7ffu) == float_to_f11(1.0f) &&
              ((words8[0] >> 11) & 0x7ffu) == float_to_f11(0.0f) &&
              ((words8[0] >> 22) & 0x3ffu) == float_to_f10(128.0f / 255.0f),
              "RGBA8 reconstruction quantizes RGB through the unsigned 11/11/10 float contract");

        auto rgba16 = std::make_shared<std::vector<uint8_t>>(8, 0);
        const float values[4] = {4.0f, 0.5f, 2.0f, 1.0f};
        for (uint32_t c = 0; c < 4; ++c) {
            const uint16_t half = float_to_half(values[c]);
            std::memcpy(rgba16->data() + c * 2, &half, sizeof(half));
        }
        LiveTargetSnapshot snapshot16{1, 1, LiveTargetPixelFormat::Rgba16Float, rgba16};
        std::vector<uint8_t> packed16(4);
        CHECK(prosper::frontend::pack_live_target_r11g11b10(
                  snapshot16, packed16.data(), packed16.size()),
              "RGBA16F renderer target reconstructs as packed R11G11B10");
        uint32_t word16 = 0;
        if (packed16.size() == sizeof(word16)) std::memcpy(&word16, packed16.data(), sizeof(word16));
        CHECK((word16 & 0x7ffu) == float_to_f11(4.0f) &&
              ((word16 >> 11) & 0x7ffu) == float_to_f11(0.5f) &&
              ((word16 >> 22) & 0x3ffu) == float_to_f10(2.0f),
              "RGBA16F reconstruction preserves HDR RGB while discarding alpha");

        auto native = std::make_shared<std::vector<uint8_t>>(std::initializer_list<uint8_t>{
            0x78, 0x56, 0x34, 0x12,
        });
        LiveTargetSnapshot native_snapshot{
            1, 1, LiveTargetPixelFormat::R11G11B10Float, native};
        std::vector<uint8_t> native_copy(4);
        CHECK(prosper::frontend::pack_live_target_r11g11b10(
                  native_snapshot, native_copy.data(), native_copy.size()) &&
              native_copy == *native,
              "native R11G11B10 renderer target remains bit-exact through CPU fallback");

        snapshot16.pixels = std::make_shared<std::vector<uint8_t>>(7, 0);
        CHECK(!prosper::frontend::pack_live_target_r11g11b10(
                  snapshot16, packed16.data(), packed16.size()),
              "R11G11B10 reconstruction rejects a malformed renderer snapshot");
    }

    // #1127: the seed-skip re-prove counter (seed_reprove.hpp). interval 0 disables (old prove-once);
    // interval N fires on the Nth fast-skip and resets, so a Full-cached data-dependent shader is
    // re-proven within N fast-skips instead of trusting a stale "covers every texel" verdict forever.
    {
        using prosper::frontend::seed_reprove_due;
        using prosper::frontend::dispatch_has_enough_threads_for_texels;
        CHECK(dispatch_has_enough_threads_for_texels(15360, 135, 1, 1920, 1080, 1),
              "vectorized/swizzled dispatch with one invocation per texel can prove coverage");
        CHECK(!dispatch_has_enough_threads_for_texels(1919, 1080, 1, 1920, 1080, 1),
              "dispatch with fewer total invocations than texels cannot prove coverage");
        CHECK(!dispatch_has_enough_threads_for_texels(0, 1080, 1, 1920, 1080, 1),
              "degenerate dispatch cannot prove coverage");
        uint32_t s = 0;
        CHECK(!seed_reprove_due(s, 0) && !seed_reprove_due(s, 0) && s == 0,
              "interval 0 never re-proves and leaves the counter untouched (prove-once)");
        s = 0;
        bool a = seed_reprove_due(s, 3), b = seed_reprove_due(s, 3), c = seed_reprove_due(s, 3);
        CHECK(!a && !b && c && s == 0, "interval 3 fires on the 3rd fast-skip and resets the counter");
        CHECK(!seed_reprove_due(s, 3) && !seed_reprove_due(s, 3) && seed_reprove_due(s, 3),
              "the re-prove cycle repeats after a reset");
        s = 0;
        CHECK(seed_reprove_due(s, 1) && seed_reprove_due(s, 1),
              "interval 1 re-proves every fast-skip (maximum soundness)");

        // #1127: the interval env-parse must fail SAFE -- garbage/overflow keeps the default rather
        // than silently disabling the safety (which an atol-style parse would, returning 0 = off).
        using prosper::frontend::seed_reprove_interval_from_env;
        CHECK(seed_reprove_interval_from_env(nullptr, 256) == 256, "unset env -> default");
        CHECK(seed_reprove_interval_from_env("", 256) == 256, "empty env -> default");
        CHECK(seed_reprove_interval_from_env("foo", 256) == 256, "non-numeric env -> default (fail-safe, not 0)");
        CHECK(seed_reprove_interval_from_env("256x", 256) == 256, "trailing junk -> default");
        CHECK(seed_reprove_interval_from_env("-4", 256) == 256, "negative -> default");
        CHECK(seed_reprove_interval_from_env("4294967296", 256) == 256, "overflow (2^32) -> default, not truncated to 0");
        CHECK(seed_reprove_interval_from_env("0", 256) == 0, "explicit 0 honored (intentionally disables re-proving)");
        CHECK(seed_reprove_interval_from_env("64", 256) == 64, "exact in-range value overrides default");
        CHECK(seed_reprove_interval_from_env("4294967295", 256) == 4294967295u, "max uint32 accepted");

        // #3285: untouched storage coverage classification. When 100% of poison survived (survived == texels),
        // zero texels were stored by the shader; it is classified as SeedCoverage::None to avoid redundant
        // seeding and writeback on every dispatch.
        using prosper::frontend::SeedCoverage;
        using prosper::frontend::classify_seed_coverage;
        using prosper::frontend::seed_coverage_name;
        CHECK(classify_seed_coverage(0, 8294400) == SeedCoverage::Full,
              "zero survivors proves Full coverage");
        CHECK(classify_seed_coverage(8294400, 8294400) == SeedCoverage::None,
              "100% survivors proves None (untouched) coverage (#3285)");
        CHECK(classify_seed_coverage(8294401, 8294400) == SeedCoverage::None,
              "clamped/overflow survivors classifies as None coverage");
        CHECK(classify_seed_coverage(1, 8294400) == SeedCoverage::Partial,
              "one survivor classifies as Partial coverage");
        CHECK(classify_seed_coverage(8294399, 8294400) == SeedCoverage::Partial,
              "all but one survivor classifies as Partial coverage");
        CHECK(std::string(seed_coverage_name(SeedCoverage::Full)).find("full-coverage") != std::string::npos,
              "Full coverage name contains full-coverage");
        CHECK(std::string(seed_coverage_name(SeedCoverage::None)).find("NONE-COVERAGE") != std::string::npos,
              "None coverage name contains NONE-COVERAGE");
        CHECK(std::string(seed_coverage_name(SeedCoverage::Partial)).find("PARTIAL-COVERAGE") != std::string::npos,
              "Partial coverage name contains PARTIAL-COVERAGE");

        // #3328 B1/N2/N3: near-full coverage classification and reprove eligibility.
        using prosper::frontend::classify_near_full_coverage;
        using prosper::frontend::seed_verdict_reprove_eligible;

        // classify_near_full_coverage:
        CHECK(classify_near_full_coverage(0, 8294400),
              "zero survivors classifies as near-full coverage");
        CHECK(classify_near_full_coverage(0, 100),
              "zero survivors on small target classifies as near-full coverage");
        // Target < 1000 texels cannot be near-full unless 0 survived
        CHECK(!classify_near_full_coverage(1, 999),
              "sub-1000 texel target with 1 survivor is not near-full");
        // 1000 texel target: 0.2% threshold is 2 texels (2 * 500 <= 1000)
        CHECK(classify_near_full_coverage(2, 1000),
              "1000 texels with 2 survivors clears 0.2% near-full bound");
        CHECK(!classify_near_full_coverage(3, 1000),
              "1000 texels with 3 survivors exceeds 0.2% near-full bound");
        // 4K target (3840x2160 = 8,294,400 texels): minimap/cutout tolerance
        CHECK(classify_near_full_coverage(16000, 8294400),
              "4K target with 16000 unwritten texels (minimap cutout ~0.19%) classifies as near-full");
        CHECK(!classify_near_full_coverage(17000, 8294400),
              "4K target with 17000 unwritten texels (>0.2%) rejects near-full");

        // seed_verdict_reprove_eligible (#3328 B1/N2):
        // Full and None verdicts are always eligible for periodic re-proving
        CHECK(seed_verdict_reprove_eligible(SeedCoverage::Full, ~0ULL, false),
              "Full coverage verdict is reprove-eligible");
        CHECK(seed_verdict_reprove_eligible(SeedCoverage::None, ~0ULL, false),
              "None coverage verdict is reprove-eligible");
        // Default Partial verdict (untouched, no active optimization) seeds every time, no re-proving needed
        CHECK(!seed_verdict_reprove_eligible(SeedCoverage::Partial, ~0ULL, false),
              "unoptimized Partial verdict does not re-prove (always seeds)");
        // Partial verdict with active layer mask optimization MUST periodically re-prove (B1)
        constexpr uint64_t kLayer0Only = 0x1ULL;
        CHECK(seed_verdict_reprove_eligible(SeedCoverage::Partial, kLayer0Only, false),
              "Partial verdict with layer mask is reprove-eligible (B1 bounds staleness)");
        // Partial verdict with active near-full coverage bypass MUST periodically re-prove (N2)
        CHECK(seed_verdict_reprove_eligible(SeedCoverage::Partial, ~0ULL, true),
              "Partial verdict with near-full coverage is reprove-eligible (N2 bounds staleness)");
        CHECK(seed_verdict_reprove_eligible(SeedCoverage::Partial, kLayer0Only, true),
              "Partial verdict with both layer mask and near-full is reprove-eligible");

        // B1 cycle verification: Partial with layer mask fires on interval and resets
        {
            uint32_t partial_skips = 0;
            const bool eligible = seed_verdict_reprove_eligible(SeedCoverage::Partial, kLayer0Only, false);
            CHECK(eligible, "layer-masked partial is eligible");
            bool fired = false;
            for (uint32_t i = 0; i < 3; ++i) {
                if (eligible && seed_reprove_due(partial_skips, 3)) fired = true;
            }
            CHECK(fired && partial_skips == 0,
                  "reprove_eligible Partial verdict fires re-proving at interval and resets counter");
        }

        // #3328 B2: the >64-layer array mask. Each arm below FAILS on the previous revision, which
        // set `written_layers = (survived < texels) ? 1 : 0` for these resources -- bit 0 as a
        // boolean. That is not a conservative approximation: the retile/pack consumers read bit
        // `layer`, so layers 1..63 were skipped as untouched while layers >= 64 fell through the
        // `layer < 64` guard and were written from non-zero-filled scratch.
        {
            using prosper::frontend::classify_array_layer_coverage;
            using prosper::frontend::array_all_layers_written;

            // A fully-written 100-layer array. Old code: written_layers == 1 (bit 0 only).
            std::vector<size_t> none_survived(100, 0);
            const auto deep_full = classify_array_layer_coverage(100, none_survived.data(), 64,
                                                                 /*survived=*/0, /*texels=*/6400);
            CHECK(deep_full.written_layers == ~0ULL,
                  "depth>64 publishes the no-masking sentinel, never a bit-0 boolean");
            CHECK(!deep_full.exact, "depth>64 mask is marked inexact");
            CHECK(!array_all_layers_written(100, deep_full.written_layers, deep_full.exact),
                  "an inexact sentinel mask must not promote coverage to Full");

            // The dangerous direction: NOTHING written on a >64-layer array. A sentinel that were
            // trusted would compare all-ones and promote an untouched surface to Full.
            std::vector<size_t> all_survived(100, 64);
            const auto deep_none = classify_array_layer_coverage(100, all_survived.data(), 64,
                                                                 /*survived=*/6400, /*texels=*/6400);
            CHECK(!array_all_layers_written(100, deep_none.written_layers, deep_none.exact),
                  "untouched depth>64 array is never promoted to Full");
            CHECK(!deep_none.any_written_partial, "untouched array reports no partial write");

            // No per-layer counts (layer_texels == 0) had the identical defect at ANY depth.
            const auto no_counts = classify_array_layer_coverage(8, nullptr, 0, 10, 100);
            CHECK(no_counts.written_layers == ~0ULL && !no_counts.exact,
                  "missing per-layer counts disable masking rather than approximate it");

            // Exactly 64 layers still works and must not use the UB `1ULL << 64`.
            std::vector<size_t> s64(64, 0);
            const auto d64 = classify_array_layer_coverage(64, s64.data(), 16, 0, 1024);
            CHECK(d64.exact && d64.written_layers == ~0ULL,
                  "depth==64 computes a real all-ones mask");
            CHECK(array_all_layers_written(64, d64.written_layers, d64.exact),
                  "depth==64 fully written is promoted");

            // Ordinary masked case still behaves: layers 0 and 2 of 4 written.
            std::vector<size_t> mixed{0, 16, 0, 16};
            const auto part = classify_array_layer_coverage(4, mixed.data(), 16, 32, 64);
            CHECK(part.exact && part.written_layers == 0b0101ULL,
                  "per-layer mask is exact below 64 layers");
            CHECK(!array_all_layers_written(4, part.written_layers, part.exact),
                  "partially written array is not promoted");

            // The ONE input where `exact` decides the answer. Below 64 the comparison is against
            // (1<<depth)-1, which ~0ULL fails anyway; above 64 the depth guard returns first. So
            // depth==64 with no per-layer counts is the only case that can catch a future edit
            // deleting the `!exact ||` guard -- and without this arm, deleting it passes everything.
            const auto d64_nocounts = classify_array_layer_coverage(64, nullptr, 0,
                                                                    /*survived=*/10, /*texels=*/1024);
            CHECK(!array_all_layers_written(64, d64_nocounts.written_layers, d64_nocounts.exact),
                  "a depth-64 sentinel mask must not promote an unmeasured surface to Full");
        }
    }

    // MinGW's lround dominates full-HD storage-image writeback. Prove the bounded integer path is
    // identical to the previous conversion across all half-float values, every UNORM threshold and
    // adjacent float, plus a deterministic million-value float32 sample.
    auto reference_unorm8 = [](uint32_t bits) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        if (std::isnan(value)) value = 0.0f;
        value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
        return static_cast<uint8_t>(std::lround(value * 255.0f));
    };
    auto check_unorm8 = [&](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return prosper::frontend::storage_pack_unorm8(bits) == reference_unorm8(bits);
    };
    bool unorm8_matches = true;
    const float edge_values[] = {
        -std::numeric_limits<float>::infinity(), -1.0f, -0.0f, 0.0f,
        std::numeric_limits<float>::denorm_min(), 1.0f,
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
    };
    for (float value : edge_values) unorm8_matches &= check_unorm8(value);
    for (uint32_t i = 0; i < 255; ++i) {
        const float threshold = (static_cast<float>(i) + 0.5f) / 255.0f;
        unorm8_matches &= check_unorm8(std::nextafter(threshold, 0.0f));
        unorm8_matches &= check_unorm8(threshold);
        unorm8_matches &= check_unorm8(std::nextafter(threshold, 1.0f));
    }
    for (uint32_t bits = 0; bits <= 0xffff; ++bits) {
        const float value = prosper::gpu::half_to_float(static_cast<uint16_t>(bits));
        unorm8_matches &= check_unorm8(value);
    }
    uint32_t random_bits = 0x6d2b79f5u;
    for (uint32_t i = 0; i < 1000000; ++i) {
        random_bits ^= random_bits << 13;
        random_bits ^= random_bits >> 17;
        random_bits ^= random_bits << 5;
        if (prosper::frontend::storage_pack_unorm8(random_bits) !=
            reference_unorm8(random_bits)) {
            unorm8_matches = false;
            break;
        }
    }
    CHECK(unorm8_matches, "fast UNORM8 pack is equivalent to the lround reference");

    auto reference_unorm16 = [](uint32_t bits) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        if (std::isnan(value)) value = 0.0f;
        value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
        return static_cast<uint16_t>(std::lround(value * 65535.0f));
    };
    bool unorm16_matches = true;
    for (uint32_t raw = 0; raw <= 0xffffu; ++raw) {
        const float value = raw / 65535.0f;
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        if (prosper::frontend::storage_pack_unorm16(bits) != raw) {
            unorm16_matches = false;
            break;
        }
    }
    for (uint32_t bits : pack_channels) {
        if (prosper::frontend::storage_pack_unorm16(bits) != reference_unorm16(bits)) {
            unorm16_matches = false;
            break;
        }
    }
    CHECK(unorm16_matches,
          "UNORM16 pack preserves all quantized channels and matches lround for float inputs");

    bool storage_unorm8_range_matches = true;
    constexpr size_t unorm_texels = pack_channels_count / 4;
    for (uint32_t components : {1u, 2u, 3u, 4u}) {
        std::vector<uint8_t> packed_unorm8(unorm_texels * components);
        prosper::frontend::storage_pack_unorm8_range(
            pack_channels.data(), components, unorm_texels, packed_unorm8.data());
        for (size_t texel = 0; texel < unorm_texels; ++texel) {
            for (uint32_t channel = 0; channel < components; ++channel) {
                const size_t source = texel * 4 + channel;
                const uint8_t expected =
                    prosper::frontend::storage_pack_unorm8(pack_channels[source]);
                const uint8_t actual = packed_unorm8[texel * components + channel];
                if (actual == expected) continue;
                std::printf("UNORM8 pack mismatch n=%u texel=%zu channel=%u f32=%08x "
                            "expected=%02x actual=%02x\n",
                            components, texel, channel, pack_channels[source], expected, actual);
                storage_unorm8_range_matches = false;
                break;
            }
            if (!storage_unorm8_range_matches) break;
        }
        if (!storage_unorm8_range_matches) break;
    }
    CHECK(storage_unorm8_range_matches,
          "packed one-, two-, three-, and four-channel UNORM8 ranges match scalar clamp and rounding");

    // Dead Cells' bound startup fill kernel, copied verbatim from eboot.elf at runtime address
    // 0x401aec200. It stores s4-s7 to record `(TGID_X << 6) + local_id_x` through the V# in s0-s3.
    static const uint32_t code[] = {
        0xd7460004, 0x04010c08, 0x7e000204, 0x7e020205, 0x7e040206,
        0x7e060207, 0xe01c2000, 0x80000004, 0xbf810000,
    };

    constexpr uint32_t records = 130;
    constexpr uint32_t launched_records = 3 * 64;
    std::vector<uint32_t> result(launched_records * 4, 0xcccccccc);
    ShaderResourceTable rt;
    ShaderResource buffer;
    // Runtime metadata classifies compute direct type-1 V#s as ConstantBuffer even though MUBUF
    // format stores use them; both classes lower to storage buffers.
    buffer.cls = ResourceClass::ConstantBuffer;
    buffer.format = DataFormat::Uint32;
    buffer.num_components = 4;
    buffer.binding = 2;
    buffer.gpu_addr = (uint64_t)(uintptr_t)result.data();
    // Deliberately expose the padded Vulkan lanes as valid writable storage. Only the generated
    // exact-thread guard can keep records 130..191 unchanged; robust buffer access cannot hide a bug.
    buffer.size = launched_records * 4 * sizeof(uint32_t);
    buffer.stride = 4 * sizeof(uint32_t);
    buffer.sgpr_base = 0;
    rt.resources.push_back(buffer);

    ComputeShaderConfig config;
    config.user_sgprs = {
        0, 0, 0, 0,
        0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00,
    };
    config.local_x = 64;
    config.exact_thread_extent = true;
    config.threads_x = records;
    config.threads_y = config.threads_z = 1;
    config.tidig_comp_cnt = 0;
    config.tgid_x_en = true;

    std::vector<uint32_t> spirv = recompile_compute(
        code, sizeof(code) / sizeof(code[0]), &rt, config);
    CHECK(!spirv.empty(), "real Dead Cells compute kernel recompiles");
    if (spirv.empty()) return 1;
    CHECK(classify_compute_cpu_fast_path(code, std::size(code)) ==
              ComputeCpuFastPath::FillSgprUvec4 &&
          classify_compute_cpu_fast_path(code, std::size(code) - 1) ==
              ComputeCpuFastPath::None,
          "buffer-fill CPU fast path requires the complete exact RDNA2 program");
    std::array<uint32_t, std::size(code)> altered_fill_code{};
    std::copy_n(code, std::size(code), altered_fill_code.begin());
    altered_fill_code[6] ^= 1u;
    CHECK(classify_compute_cpu_fast_path(altered_fill_code.data(), altered_fill_code.size()) ==
              ComputeCpuFastPath::None,
          "buffer-fill CPU fast path rejects a one-bit instruction change");
    ComputeShaderConfig alternate_config = config;
    alternate_config.user_sgprs[4] ^= 0xffffffffu;
    alternate_config.user_sgprs[7] ^= 0x13579bdfu;
    CHECK(recompile_compute(code, sizeof(code) / sizeof(code[0]), &rt, alternate_config) == spirv,
          "per-dispatch user SGPR values do not specialize the reusable SPIR-V module");

    auto report = validate_spirv_descriptor_interface(
        spirv, &rt, 0, SpirvShaderStage::Compute);
    for (const auto& issue : report.issues) {
        if (issue.error)
            std::printf("descriptor error: %s binding=%u expected=%s actual=%s\n",
                        descriptor_issue_name(issue.code), issue.binding,
                        spirv_descriptor_kind_name(issue.expected),
                        spirv_descriptor_kind_name(issue.actual));
    }
    CHECK(report.ok(), "real compute descriptor interface validates");

    // --- a partial workgroup that also uses a barrier -------------------------------------------
    //
    // The kernel above already ends in a partial workgroup (130 records launched as 3x64), and the
    // divergent entry guard is what keeps its padded lanes from writing. Add one s_barrier and that
    // guard becomes illegal: the padded invocations would skip the body, and with it the barrier
    // their peers block on. Twelve of GTA V's compute dispatches were dropped for exactly this
    // (`reason=partial-workgroup-barrier threads=1143x1x1 local=64x1x1`).
    //
    // The correct lowering already existed -- split the stream at its barriers, compile each
    // barrier-free phase through the dispatcher's per-lane ACTIVE bit, and emit each barrier from
    // the outer shell where every invocation reaches it. It was gated behind `branch_count > 2`,
    // which excluded precisely the programs that satisfy its soundness proof most easily.
    {
        auto count_opcode = [](const std::vector<uint32_t>& words, uint16_t opcode) {
            size_t count = 0;
            for (size_t at = 5; at < words.size();) {
                const uint32_t length = words[at] >> 16;
                if (length == 0) break;
                count += static_cast<uint16_t>(words[at]) == opcode;
                at += length;
            }
            return count;
        };
        constexpr uint16_t kOpSwitch = 251, kOpControlBarrier = 224;
        // SOPP opcode 0x0a (SoppOpcode::Barrier) at the SOPP encoding base 0xbf800000.
        constexpr uint32_t kSBarrier = 0xbf8a0000u;
        std::vector<uint32_t> barrier_code(code, code + std::size(code));
        // Ahead of the store, so the guest memory effect sits on the far side of the barrier and the
        // program genuinely splits into two non-empty phases.
        barrier_code.insert(barrier_code.end() - 3, kSBarrier);

        const std::vector<uint32_t> barrier_spirv =
            recompile_compute(barrier_code.data(), barrier_code.size(), &rt, config);
        CHECK(!barrier_spirv.empty(),
              "a partial workgroup that uses a barrier recompiles instead of being rejected");
        // Guards against a vacuous pass: were the barrier dropped in translation, the module would
        // compile for the wrong reason and every assertion below would still hold.
        // Measured: 5 barriers for one guest s_barrier. One is the guest's, emitted by the phase
        // shell at function scope; the other four are the two dispatchers' own synchronized common
        // phases. Those are uniform by construction -- a padded invocation stays inside the
        // dispatcher loop and reaches them, and only its switch selector differs -- which is the
        // same property that makes ACTIVE=false safe in the first place. So the count is not the
        // interesting number here; that it is non-zero rules out the barrier having been dropped.
        CHECK(count_opcode(barrier_spirv, kOpControlBarrier) > 0,
              "the recompiled module really does contain the barrier under test");
        // THE discriminator, and the reason this is asserted by structure rather than by exit code.
        // Two lowerings both produce a module that compiles and contains a barrier:
        //   sound -- split at the barrier, one dispatcher per phase, barrier between them at
        //            function scope, reached by every invocation: TWO OpSwitch.
        //   unsound -- one dispatcher over the whole stream with the barrier inside a switch case,
        //            reached only by invocations whose ACTIVE bit is set: ONE OpSwitch, and a
        //            latent deadlock that no exit code, and no spirv-val run, would report.
        // Counting them is what tells those two apart.
        CHECK(count_opcode(barrier_spirv, kOpSwitch) == 2,
              "the barrier splits the program into two dispatched phases rather than sitting inside "
              "one switch case, where only active invocations would reach it");

        const std::vector<uint32_t> guarded_spirv =
            recompile_compute(code, std::size(code), &rt, config);
        CHECK(!guarded_spirv.empty() && count_opcode(guarded_spirv, kOpControlBarrier) == 0 &&
                  count_opcode(guarded_spirv, kOpSwitch) == 0,
              "the same program without the barrier still takes the cheap entry guard, so the fix "
              "did not route every partial workgroup through a dispatcher");

        // A complete workgroup has no padded lane to suppress, so a barrier costs it nothing and
        // this was never the rejected case -- the arm that keeps the claim about WHY it was rejected
        // honest, rather than merely about the barrier being present.
        ComputeShaderConfig exact_config = config;
        exact_config.threads_x = 192;
        CHECK(!recompile_compute(barrier_code.data(), barrier_code.size(), &rt,
                                 exact_config).empty(),
              "a barrier in a workgroup with no partial tail was never the rejected case");
    }

    ComputeItem item;
    item.spirv = spirv;
    item.user_sgprs = config.user_sgprs;
    item.resources = std::make_shared<ShaderResourceTable>(rt);
    item.launch.threads_x = records;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_x = 64;
    item.launch.groups_x = 3;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_y = item.launch.groups_z = 1;
    item.code_addr = 0x401aec200;
    item.dispatch_index = 7;
    item.submit_no = 11;
    item.command_order = 70;
    item.recompile_config = config;
    item.recompile_config_available = true;

    // GTA V reaches two compute programs whose only external operations use fully-known RAW V#s
    // with NUM_RECORDS=0. Production discovery materializes an exact-PC marker and recompilation
    // removes the buffer access entirely, so reflection correctly reports no descriptor. The live
    // backend must recognize only that complete proof as successful without submitting empty work.
    const uint32_t zero_record_code[] = {
        0xE0300000u, 0x80000000u, // buffer_load_dword v0, off, s[0:3]
        0xBF810000u,
    };
    const uint32_t zero_record_seed[4] = {
        0x00000000u, 0x00100000u, 0x00000000u, 0x00016204u,
    };
    ShaderResourceTable zero_record_rt;
    const std::vector<SrtUse> zero_record_uses = add_compute_buffer_resources(
        zero_record_rt, zero_record_code, std::size(zero_record_code), zero_record_seed,
        std::size(zero_record_seed));
    assign_convention_bindings(zero_record_rt, 2);
    ComputeShaderConfig zero_record_config;
    zero_record_config.user_sgprs.assign(zero_record_seed,
                                         zero_record_seed + std::size(zero_record_seed));
    zero_record_config.local_x = zero_record_config.local_y = zero_record_config.local_z = 1;
    const std::vector<uint32_t> zero_record_spirv = recompile_compute(
        zero_record_code, std::size(zero_record_code), &zero_record_rt,
        zero_record_config);
    const DescriptorValidationReport zero_record_report = validate_spirv_descriptor_interface(
        zero_record_spirv, &zero_record_rt, 0, SpirvShaderStage::Compute, false);
    CHECK(zero_record_uses.size() == 1 && zero_record_rt.resources.size() == 1 &&
              is_zero_record_raw_buffer(zero_record_rt.resources[0]) &&
              !zero_record_spirv.empty() && zero_record_report.ok() &&
              zero_record_report.descriptors.empty(),
          "production zero-record RAW program recompiles with an exact marker and no descriptors");

    ComputeItem zero_record_item;
    zero_record_item.spirv = zero_record_spirv;
    zero_record_item.user_sgprs = zero_record_config.user_sgprs;
    zero_record_item.resources = std::make_shared<ShaderResourceTable>(zero_record_rt);
    zero_record_item.launch.threads_x = zero_record_item.launch.threads_y =
        zero_record_item.launch.threads_z = 1;
    zero_record_item.launch.local_x = zero_record_item.launch.local_y =
        zero_record_item.launch.local_z = 1;
    zero_record_item.launch.groups_x = zero_record_item.launch.groups_y =
        zero_record_item.launch.groups_z = 1;
    zero_record_item.code_addr = 0x413cf4900ull;
    zero_record_item.dispatch_index = 6;
    zero_record_item.submit_no = 11;
    zero_record_item.command_order = 69;
    zero_record_item.recompile_config = zero_record_config;
    zero_record_item.recompile_config_available = true;

    const uint64_t zero_record_submits_before =
        prosper::frontend::live_compute_queue_submit_attempts();
    CHECK(prosper::frontend::execute_live_compute_items({zero_record_item}),
          "all-marker descriptorless compute completes as a proven live-backend no-op");
    ComputeItem forged_guarded_store_item = zero_record_item;
    forged_guarded_store_item.resources = std::make_shared<ShaderResourceTable>();
    ShaderResource guarded_store_resource = zero_record_rt.resources[0];
    guarded_store_resource.stride = kProvenNullGuardedRawStoreStride;
    guarded_store_resource.fetch_pc = 74u;
    forged_guarded_store_item.resources->resources.push_back(guarded_store_resource);
    CHECK(is_proven_null_guarded_raw_store(
              forged_guarded_store_item.resources->resources.front()) &&
              !prosper::frontend::execute_live_compute_items({forged_guarded_store_item}),
          "a forged guarded-null-store marker has no live-backend proof token");

    std::array<uint8_t, kGtaNullableOutputWitnessBytes> nullable_output_witness{};
    ShaderResource nullable_output_resource = zero_record_rt.resources[0];
    nullable_output_resource.gpu_addr = 0x720000u;
    nullable_output_resource.size = kGtaNullableOutputWitnessBytes;
    nullable_output_resource.stride = kProvenNullNullableRawBufferStride;
    nullable_output_resource.fetch_pc = 38u;
    nullable_output_resource.host_data = nullable_output_witness.data();
    nullable_output_resource.host_data_size = nullable_output_witness.size();
    ComputeItem nullable_output_item = zero_record_item;
    nullable_output_item.resources = std::make_shared<ShaderResourceTable>();
    nullable_output_item.resources->resources.push_back(nullable_output_resource);
    CHECK(is_proven_null_nullable_raw_buffer(nullable_output_resource) &&
              !prosper::frontend::execute_live_compute_items({nullable_output_item}),
          "a nullable-output marker has no live-backend authority without its separate token");
    nullable_output_item.nullable_output_raw_buffer_validated = true;
    CHECK(prosper::frontend::execute_live_compute_items({nullable_output_item}),
          "a validated nullable-output marker completes as a descriptorless live-backend no-op");
    ComputeItem malformed_nullable_output_item = nullable_output_item;
    malformed_nullable_output_item.resources =
        std::make_shared<ShaderResourceTable>(*nullable_output_item.resources);
    malformed_nullable_output_item.resources->resources[0].fetch_pc = UINT32_MAX;
    CHECK(!prosper::frontend::execute_live_compute_items({malformed_nullable_output_item}),
          "the nullable-output token cannot authorize a malformed impossible-stride marker");

    auto cf9200_root_witness =
        prosper::test::gta5_cf9200_source_and_output_null_root();
    ShaderResource cf9200_marker = zero_record_rt.resources[0];
    cf9200_marker.gpu_addr = 0x730000u;
    cf9200_marker.size = kGtaCf9200RootBytes;
    cf9200_marker.stride = kGtaCf9200NoBackingStride;
    cf9200_marker.fetch_pc = 5u;
    cf9200_marker.host_data =
        reinterpret_cast<uint8_t*>(cf9200_root_witness.data());
    cf9200_marker.host_data_size = sizeof(cf9200_root_witness);
    ComputeItem cf9200_marker_item = zero_record_item;
    cf9200_marker_item.resources = std::make_shared<ShaderResourceTable>();
    cf9200_marker_item.resources->resources.push_back(cf9200_marker);
    CHECK(is_proven_gta5_cf9200_no_backing(cf9200_marker) &&
              !prosper::frontend::execute_live_compute_items({cf9200_marker_item}),
          "a GTA root-record marker has no live-backend authority without its token");
    cf9200_marker_item.gta5_cf9200_no_backing_validated = true;
    CHECK(prosper::frontend::execute_live_compute_items({cf9200_marker_item}),
          "a validated GTA root-record marker completes as a descriptorless no-op");
    ComputeItem malformed_cf9200_item = cf9200_marker_item;
    malformed_cf9200_item.resources =
        std::make_shared<ShaderResourceTable>(*cf9200_marker_item.resources);
    malformed_cf9200_item.resources->resources[0].size--;
    CHECK(!prosper::frontend::execute_live_compute_items({malformed_cf9200_item}),
          "the root-record token cannot authorize a malformed marker");

    std::array<uint32_t, 81> guarded_store_code;
    guarded_store_code.fill(0xbf800000u); // s_nop 0
    guarded_store_code[22] = 0xbe92047eu; // s_mov_b64 s[18:19], exec
    guarded_store_code[41] = 0xbefe0412u; // s_mov_b64 exec, s[18:19]
    guarded_store_code[42] = 0xbf128002u;
    guarded_store_code[43] = 0x7d8a00f9u;
    guarded_store_code[44] = 0x06868080u;
    guarded_store_code[45] = 0x85ea8012u;
    guarded_store_code[46] = 0x8dea006au;
    guarded_store_code[47] = 0x87fe126au;
    guarded_store_code[48] = 0xbf88001fu;
    guarded_store_code[74] = 0xe0740030u;
    guarded_store_code[75] = 0x80000700u;
    guarded_store_code[80] = 0xbf810000u;
    ComputeShaderConfig guarded_store_config;
    guarded_store_config.user_sgprs.resize(4u);
    guarded_store_config.local_x = 64u;
    ShaderResourceTable guarded_store_table;
    guarded_store_table.resources.push_back(guarded_store_resource);
    ComputeItem guarded_store_item = zero_record_item;
    guarded_store_item.spirv = recompile_compute(
        guarded_store_code.data(), guarded_store_code.size(),
        &guarded_store_table, guarded_store_config);
    guarded_store_item.user_sgprs = guarded_store_config.user_sgprs;
    guarded_store_item.recompile_config = guarded_store_config;
    guarded_store_item.resources =
        std::make_shared<ShaderResourceTable>(guarded_store_table);
    guarded_store_item.null_guarded_raw_store_validated =
        !guarded_store_item.spirv.empty() &&
        rdna2_gta5_null_guarded_raw_store_dispatch(
            guarded_store_code.data(), guarded_store_code.size(),
            guarded_store_config.user_sgprs.data(),
            guarded_store_config.user_sgprs.size());
    CHECK(guarded_store_item.null_guarded_raw_store_validated &&
              prosper::frontend::execute_live_compute_items({guarded_store_item}),
          "validated guarded-null stores complete as a live-backend no-op");
    ComputeItem mixed_marker_item = guarded_store_item;
    mixed_marker_item.resources = std::make_shared<ShaderResourceTable>(zero_record_rt);
    mixed_marker_item.resources->resources.push_back(guarded_store_resource);
    CHECK(prosper::frontend::execute_live_compute_items({mixed_marker_item}),
          "mixed zero-record and validated guarded-store markers retain descriptorless success");
    ComputeItem empty_zero_record_item = zero_record_item;
    empty_zero_record_item.resources = std::make_shared<ShaderResourceTable>();
    CHECK(!prosper::frontend::execute_live_compute_items({empty_zero_record_item}),
          "descriptorless compute with an empty runtime table remains fail-visible");
    ComputeItem ordinary_zero_record_item = zero_record_item;
    ordinary_zero_record_item.resources = std::make_shared<ShaderResourceTable>();
    ShaderResource unused_ordinary_resource;
    unused_ordinary_resource.cls = ResourceClass::ConstantBuffer;
    unused_ordinary_resource.format = DataFormat::Uint32;
    unused_ordinary_resource.num_components = 1;
    unused_ordinary_resource.binding = 2;
    unused_ordinary_resource.gpu_addr = reinterpret_cast<uint64_t>(result.data());
    unused_ordinary_resource.size = sizeof(uint32_t);
    ordinary_zero_record_item.resources->resources.push_back(unused_ordinary_resource);
    CHECK(!prosper::frontend::execute_live_compute_items({ordinary_zero_record_item}),
          "descriptorless compute with an unused ordinary resource remains fail-visible");
    CHECK(prosper::frontend::live_compute_queue_submit_attempts() ==
              zero_record_submits_before,
          "proven and rejected descriptorless programs never attempt a Vulkan queue submit");

    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "production live backend executes the game kernel");
    CHECK(result.size() == launched_records * 4, "compute resource retains its padded declared size");
    const uint32_t expected[4] = {0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00};
    bool all_filled = result.size() == launched_records * 4;
    for (uint32_t record = 0; record < records; record++) {
        for (uint32_t component = 0; component < 4; component++) {
            if (result[record * 4 + component] != expected[component]) {
                std::printf("FAIL: record %u component %u = %08x, expected %08x\n",
                            record, component, result[record * 4 + component], expected[component]);
                all_filled = false;
                break;
            }
        }
        if (!all_filled) break;
    }
    CHECK(all_filled, "all 130 records are filled across three workgroups");
    bool padded_lanes_untouched = true;
    for (uint32_t record = records; record < launched_records; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            padded_lanes_untouched &= result[record * 4 + component] == 0xcccccccc;
    CHECK(padded_lanes_untouched,
          "partial workgroup suppresses all 62 padded invocations without a guest bounds check");

    // GTA V's culling programs scan the physical high and low dwords of a saved Wave64 predicate.
    // Their live route cannot require a matching Vulkan subgroup width, so execute the exact FFBH
    // packets through the portable arbitrary-CFG dispatcher. The two masks prove both paths: lanes
    // 0..16 force the zero high dword and return 15 from the low dword; lane 63 returns zero directly
    // from the high dword. Crossing scalar branches keep this on the same dispatcher path as GTA.
    {
        std::array<uint32_t, 13> portable_mask_ffbh_code = {
            0x7d8800f9u, 0x06868891u,            // pc0: s[8:9] = ballot(17 > local_id.x)
            0xbf060000u,                         // pc2: establish SCC
            0xbf840003u,                         // pc3: s_cbranch_scc0 -> pc7
            0x7e147209u,                         // pc4: exact v_ffbh_u32 v10,s9
            0x7d8a14c1u,                         // pc5: high result != -1
            0xbf860002u,                         // pc6: s_cbranch_vccz -> pc9
            0xbf060000u,                         // pc7: independent SCC lifetime
            0xbf850001u,                         // pc8: s_cbranch_scc1 -> pc10
            0x7e147208u,                         // pc9: exact v_ffbh_u32 v10,s8
            0xe0702000u, 0x80000a00u,            // pc10: buffer_store_dword v10,v0,s[0:3],idxen
            0xbf810000u,                         // pc12: s_endpgm
        };
        std::vector<uint32_t> mask_ffbh_result(64, 0xccccccccu);
        ShaderResourceTable mask_ffbh_rt;
        ShaderResource mask_ffbh_buffer;
        mask_ffbh_buffer.cls = ResourceClass::ConstantBuffer;
        mask_ffbh_buffer.format = DataFormat::Uint32;
        mask_ffbh_buffer.num_components = 1;
        mask_ffbh_buffer.binding = 2;
        mask_ffbh_buffer.gpu_addr =
            reinterpret_cast<uint64_t>(mask_ffbh_result.data());
        mask_ffbh_buffer.size = mask_ffbh_result.size() * sizeof(uint32_t);
        mask_ffbh_buffer.stride = sizeof(uint32_t);
        mask_ffbh_buffer.sgpr_base = 0;
        mask_ffbh_rt.resources.push_back(mask_ffbh_buffer);

        ComputeShaderConfig mask_ffbh_config;
        mask_ffbh_config.user_sgprs.resize(4);
        mask_ffbh_config.local_x = 64;
        mask_ffbh_config.wave_size = 64;
        mask_ffbh_config.tidig_comp_cnt = 0;
        mask_ffbh_config.native_subgroup_size = 0;

        auto run_mask_ffbh = [&](uint32_t producer0, uint32_t producer1,
                                 uint32_t expected, uint64_t code_addr) {
            portable_mask_ffbh_code[0] = producer0;
            portable_mask_ffbh_code[1] = producer1;
            std::fill(mask_ffbh_result.begin(), mask_ffbh_result.end(), 0xccccccccu);
            const std::vector<uint32_t> module = recompile_compute(
                portable_mask_ffbh_code.data(), portable_mask_ffbh_code.size(),
                &mask_ffbh_rt, mask_ffbh_config);
            CHECK(!module.empty(),
                  "portable Wave64 saved-mask FFBH execution fixture recompiles");
            if (module.empty()) return false;

            ComputeItem mask_ffbh_item;
            mask_ffbh_item.spirv = module;
            mask_ffbh_item.user_sgprs = mask_ffbh_config.user_sgprs;
            mask_ffbh_item.resources =
                std::make_shared<ShaderResourceTable>(mask_ffbh_rt);
            mask_ffbh_item.launch.threads_x = 64;
            mask_ffbh_item.launch.threads_y = mask_ffbh_item.launch.threads_z = 1;
            mask_ffbh_item.launch.local_x = 64;
            mask_ffbh_item.launch.local_y = mask_ffbh_item.launch.local_z = 1;
            mask_ffbh_item.launch.groups_x = mask_ffbh_item.launch.groups_y =
                mask_ffbh_item.launch.groups_z = 1;
            mask_ffbh_item.code_addr = code_addr;
            mask_ffbh_item.recompile_config = mask_ffbh_config;
            mask_ffbh_item.recompile_config_available = true;
            const bool executed =
                prosper::frontend::execute_live_compute_items({mask_ffbh_item});
            CHECK(executed, "portable Wave64 saved-mask FFBH executes on Vulkan");
            return executed && std::all_of(
                mask_ffbh_result.begin(), mask_ffbh_result.end(),
                [&](uint32_t value) { return value == expected; });
        };

        CHECK(run_mask_ffbh(
                  0x7d8800f9u, 0x06868891u, 15u, 0x413e15400ull),
              "portable Wave64 FFBH assembles and scans the saved low mask dword");
        CHECK(run_mask_ffbh(
                  0x7d8400f9u, 0x068688bfu, 0u, 0x413e16400ull),
              "portable Wave64 FFBH assembles and scans the saved high mask dword");
    }

    std::fill(result.begin(), result.end(), 0xddddddddu);
    ComputeItem cpu_fill_item = item;
    cpu_fill_item.cpu_fast_path = ComputeCpuFastPath::FillSgprUvec4;
    uint32_t cpu_fill_write_notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char*) {
        if (addr == buffer.gpu_addr && size == buffer.size)
            ++cpu_fill_write_notifications;
    });
    std::vector<ComputeAuthorityBoundary> cpu_fill_authority_boundaries;
    bool cpu_fill_boundary_preceded_write = false;
    set_compute_authority_boundary_observer(
        [&](const ComputeAuthorityBoundary& boundary) {
            cpu_fill_authority_boundaries.push_back(boundary);
            cpu_fill_boundary_preceded_write = !result.empty() &&
                result.front() == 0xddddddddu;
        });
    const uint64_t cpu_fills_before =
        prosper::frontend::live_compute_cpu_fill_dispatches();
    CHECK(prosper::frontend::execute_live_compute_items({cpu_fill_item}),
          "exact buffer-fill program executes through the CPU fast path");
    set_compute_authority_boundary_observer({});
    set_guest_gpu_write_observer({});
    bool cpu_fill_matches = true;
    for (uint32_t record = 0; record < records && cpu_fill_matches; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            cpu_fill_matches &= result[record * 4 + component] == expected[component];
    bool cpu_fill_padding_untouched = true;
    for (uint32_t record = records; record < launched_records; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            cpu_fill_padding_untouched &= result[record * 4 + component] == 0xddddddddu;
    CHECK(cpu_fill_matches && cpu_fill_padding_untouched &&
              cpu_fill_write_notifications == 1 &&
              prosper::frontend::live_compute_cpu_fill_dispatches() ==
                  cpu_fills_before + 1 && cpu_fill_boundary_preceded_write &&
              cpu_fill_authority_boundaries.size() == 1 &&
              cpu_fill_authority_boundaries[0].kind ==
                  ComputeAuthorityBoundaryKind::Compute &&
              cpu_fill_authority_boundaries[0].range_known &&
              cpu_fill_authority_boundaries[0].address == buffer.gpu_addr &&
              cpu_fill_authority_boundaries[0].bytes == records * 16u,
          "CPU fill matches Vulkan, preserves padded lanes, and invalidates the declared range");

    ShaderResourceTable narrow_stride_rt = rt;
    narrow_stride_rt.resources[0].stride = 8;
    const std::vector<uint32_t> narrow_stride_spirv = recompile_compute(
        code, std::size(code), &narrow_stride_rt, config);
    CHECK(!narrow_stride_spirv.empty(), "alternate-stride fill kernel recompiles");
    if (!narrow_stride_spirv.empty()) {
        std::fill(result.begin(), result.end(), 0xabababab);
        ComputeItem narrow_stride_item = cpu_fill_item;
        narrow_stride_item.spirv = narrow_stride_spirv;
        narrow_stride_item.resources =
            std::make_shared<ShaderResourceTable>(narrow_stride_rt);
        const uint64_t narrow_stride_fills_before =
            prosper::frontend::live_compute_cpu_fill_dispatches();
        CHECK(prosper::frontend::execute_live_compute_items({narrow_stride_item}),
              "noncanonical fill descriptor falls back to Vulkan");
        CHECK(result[300] == 0xabababab &&
                  prosper::frontend::live_compute_cpu_fill_dispatches() ==
                      narrow_stride_fills_before,
              "CPU fast path does not replace the descriptor's eight-byte stride semantics");
    }

    ComputeShaderConfig extra_user_config = config;
    extra_user_config.user_sgprs.push_back(0);
    const std::vector<uint32_t> extra_user_spirv = recompile_compute(
        code, std::size(code), &rt, extra_user_config);
    CHECK(!extra_user_spirv.empty(), "alternate user-SGPR fill kernel recompiles");
    if (!extra_user_spirv.empty()) {
        std::fill(result.begin(), result.end(), 0xcdcdcdcdu);
        ComputeItem extra_user_item = cpu_fill_item;
        extra_user_item.spirv = extra_user_spirv;
        extra_user_item.user_sgprs = extra_user_config.user_sgprs;
        extra_user_item.recompile_config = extra_user_config;
        const uint64_t extra_user_fills_before =
            prosper::frontend::live_compute_cpu_fill_dispatches();
        CHECK(prosper::frontend::execute_live_compute_items({extra_user_item}),
              "shifted TGID input fill falls back to Vulkan");
        CHECK(result[100 * 4] == 0xcdcdcdcdu &&
                  prosper::frontend::live_compute_cpu_fill_dispatches() ==
                      extra_user_fills_before,
              "CPU fast path requires TGID.x to occupy the shader's exact s8 input");
    }

    uint32_t unchanged_write_notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char*) {
        if (addr == buffer.gpu_addr && size == buffer.size)
            unchanged_write_notifications++;
    });
    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "production live backend repeats an idempotent buffer dispatch");
    set_guest_gpu_write_observer({});
    CHECK(unchanged_write_notifications == 1,
          "idempotent compute writes still invalidate divergent renderer-resident state");

    // Astro Bot's world-map visibility kernel lowers IMAGE_ATOMIC_ADD on a tiled R32_UINT image
    // to a linear SSBO atomic. The descriptor's declared size is the logical texel count, while a
    // Sw64KbRX backing includes whole-tile padding and can therefore be larger. Replay provides that
    // physical capture through host_data_size; rejecting it against the logical size skipped the
    // entire visibility pass.
    {
        static const uint32_t image_atomic_add[] = {
            0x7e000280u, 0x7e020280u, 0x7e120281u,
            0xf0442108u, 0x00000900u, 0xbf810000u,
        };
        constexpr uint32_t atomic_width = 2048;
        constexpr uint32_t atomic_height = 64;
        constexpr uint32_t atomic_tile = static_cast<uint32_t>(TileMode::Sw64KbRX);
        constexpr size_t atomic_logical_bytes =
            static_cast<size_t>(atomic_width) * atomic_height * sizeof(uint32_t);
        const size_t atomic_guest_bytes = tiled_surface_bytes(
            atomic_width, atomic_height, atomic_tile, 0, sizeof(uint32_t));
        CHECK(atomic_guest_bytes > atomic_logical_bytes,
              "world-map atomic image fixture includes physical tile padding");
        std::vector<uint8_t> atomic_guest(atomic_guest_bytes, 0);

        ShaderResourceTable atomic_rt;
        ShaderResource atomic_image;
        atomic_image.cls = ResourceClass::StorageImage;
        atomic_image.format = DataFormat::Uint32;
        atomic_image.num_components = 1;
        atomic_image.binding = 4;
        atomic_image.sgpr_base = 0;
        atomic_image.img_dim = 1;
        atomic_image.width = atomic_width;
        atomic_image.height = atomic_height;
        atomic_image.depth = 1;
        atomic_image.tile_mode = atomic_tile;
        atomic_image.size = static_cast<uint32_t>(atomic_logical_bytes);
        atomic_image.gpu_addr = reinterpret_cast<uint64_t>(atomic_guest.data());
        atomic_image.host_data = atomic_guest.data();
        atomic_image.host_data_size = atomic_guest.size();
        atomic_rt.resources.push_back(atomic_image);

        ComputeShaderConfig atomic_config;
        atomic_config.local_x = 1;
        const std::vector<uint32_t> atomic_spirv = recompile_compute(
            image_atomic_add, std::size(image_atomic_add), &atomic_rt, atomic_config);
        CHECK(!atomic_spirv.empty(), "world-map tiled image-atomic kernel recompiles");
        if (!atomic_spirv.empty()) {
            ComputeItem atomic_item;
            atomic_item.spirv = atomic_spirv;
            atomic_item.resources = std::make_shared<ShaderResourceTable>(atomic_rt);
            atomic_item.launch.threads_x = atomic_item.launch.threads_y =
                atomic_item.launch.threads_z = 1;
            atomic_item.launch.local_x = atomic_item.launch.local_y =
                atomic_item.launch.local_z = 1;
            atomic_item.launch.groups_x = atomic_item.launch.groups_y =
                atomic_item.launch.groups_z = 1;
            atomic_item.code_addr = 0x500525200;

            uint64_t atomic_notified_bytes = 0;
            set_guest_gpu_write_observer([&](uint64_t addr, uint64_t bytes, const char*) {
                if (addr == atomic_image.gpu_addr) atomic_notified_bytes = bytes;
            });
            CHECK(prosper::frontend::execute_live_compute_items({atomic_item}),
                  "world-map tiled image-atomic dispatch accepts its physical backing");
            set_guest_gpu_write_observer({});

            std::vector<uint32_t> atomic_linear(atomic_width * atomic_height, 0);
            detile_surface(reinterpret_cast<uint8_t*>(atomic_linear.data()), atomic_guest.data(),
                           atomic_width, atomic_height, atomic_tile, 0, sizeof(uint32_t));
            CHECK(atomic_linear[0] == 1,
                  "world-map tiled image-atomic dispatch publishes its atomic result");
            CHECK(atomic_notified_bytes == atomic_guest_bytes,
                  "world-map tiled image-atomic write invalidates the full physical backing");
        }
    }

    // Plucky Squire binds the same 32 MiB writable lighting buffer to consecutive kernels even
    // when their output is byte-identical to the previous frame. Exercise the production cache at
    // its one-MiB retention threshold: the second dispatch can use the exact GPU-side baseline,
    // while a later external guest mutation must defeat that shortcut and be repaired normally.
    {
        constexpr uint32_t large_buffer_bytes = 1u << 20;
        constexpr size_t large_words = large_buffer_bytes / sizeof(uint32_t);
        std::vector<uint32_t> large_result_storage;
        uint32_t* large_result = nullptr;
#if defined(__linux__)
        void* large_mapping = mmap(nullptr, large_buffer_bytes, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(large_mapping != MAP_FAILED, "map large writable compute-buffer regression range");
        if (large_mapping == MAP_FAILED) return fails ? fails : 1;
        large_result = static_cast<uint32_t*>(large_mapping);
        prosper::host::guest_write_watch_set_fault_onstack(true);
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(large_result), large_buffer_bytes, 0x6b0000,
            0x3 /* SCE CPU_READ|CPU_WRITE */);
#else
        large_result_storage.resize(large_words);
        large_result = large_result_storage.data();
#endif
#if defined(__linux__)
        // The ordinary CPU-comparison fallback is used below the persistent-buffer threshold. It
        // also has an exact unchanged result and must use the byte-preserving notification, not
        // merely the retained GPU-comparator path exercised below.
        std::fill_n(large_result, buffer.size / sizeof(uint32_t), 0xccccccccu);
        ShaderResource watched_small_buffer = buffer;
        watched_small_buffer.gpu_addr = reinterpret_cast<uint64_t>(large_result);
        ShaderResourceTable watched_small_rt;
        watched_small_rt.resources.push_back(watched_small_buffer);
        ComputeItem watched_small_item = item;
        watched_small_item.resources = std::make_shared<ShaderResourceTable>(watched_small_rt);
        CHECK(prosper::frontend::execute_live_compute_items({watched_small_item}),
              "small writable buffer establishes its CPU-comparison result");
        auto unchanged_small_watch = prosper::host::GuestWriteWatch::create(
            reinterpret_cast<uint64_t>(large_result), buffer.size);
        CHECK(prosper::frontend::execute_live_compute_items({watched_small_item}),
              "small writable buffer repeats through CPU comparison");
        CHECK(static_cast<bool>(unchanged_small_watch) &&
                  unchanged_small_watch.query() ==
                      prosper::host::GuestWriteWatchQuery::Unchanged,
              "CPU-identical production dispatch keeps guest-byte watches clean");
        unchanged_small_watch.reset();
#endif
        std::fill_n(large_result, large_words, 0xababababu);
        ShaderResource large_buffer = buffer;
        large_buffer.gpu_addr = reinterpret_cast<uint64_t>(large_result);
        large_buffer.size = large_buffer_bytes;
        ShaderResourceTable large_rt;
        large_rt.resources.push_back(large_buffer);
        ComputeItem large_item = item;
        large_item.resources = std::make_shared<ShaderResourceTable>(large_rt);
        large_item.code_addr = 0x401aec210;

        CHECK(prosper::frontend::execute_live_compute_items({large_item}),
              "large writable buffer dispatch establishes an exact retained result");
        const std::vector<uint32_t> large_expected(large_result, large_result + large_words);
        const uint64_t buffer_skips_before =
            prosper::frontend::live_compute_buffer_gpu_result_skips();
#if defined(__linux__)
        auto unchanged_result_watch = prosper::host::GuestWriteWatch::create(
            reinterpret_cast<uint64_t>(large_result), large_buffer_bytes);
        CHECK(static_cast<bool>(unchanged_result_watch) &&
                  unchanged_result_watch.query() ==
                      prosper::host::GuestWriteWatchQuery::Unchanged,
              "arm guest-byte watch around retained compute-buffer result");
#endif
        uint32_t large_repeat_notifications = 0;
        set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char*) {
            if (addr == large_buffer.gpu_addr && size == large_buffer.size)
                ++large_repeat_notifications;
        });
        CHECK(prosper::frontend::execute_live_compute_items({large_item}),
              "large writable buffer repeats against its exact GPU baseline");
        set_guest_gpu_write_observer({});
        CHECK(std::equal(large_result, large_result + large_words, large_expected.begin()) &&
                  large_repeat_notifications == 1,
              "GPU-identical buffer output preserves bytes and architectural invalidation");
        CHECK(prosper::frontend::live_compute_buffer_gpu_result_skips() > buffer_skips_before,
              "large idempotent buffer takes the exact GPU result-comparison fast path");
#if defined(__linux__)
        CHECK(unchanged_result_watch.query() ==
                  prosper::host::GuestWriteWatchQuery::Unchanged,
              "GPU-identical production dispatch keeps guest-byte watches clean");
        unchanged_result_watch.reset();
#endif

        large_result[0] ^= 0xffffffffu;
        uint32_t large_repair_notifications = 0;
        set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char*) {
            if (addr == large_buffer.gpu_addr && size == large_buffer.size)
                ++large_repair_notifications;
        });
        CHECK(prosper::frontend::execute_live_compute_items({large_item}),
              "externally changed buffer reruns with exact source validation");
        set_guest_gpu_write_observer({});
        CHECK(std::equal(large_result, large_result + large_words, large_expected.begin()) &&
                  large_repair_notifications == 1,
              "external buffer mutation forces exact guest repair and invalidation");
#if defined(__linux__)
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(large_result), large_buffer_bytes);
        prosper::host::guest_write_watch_set_fault_onstack(false);
        munmap(large_result, large_buffer_bytes);
#endif
    }

    // #3032: the compare-flag readback stride had no regression arm because nothing in this suite
    // ever reached two GPU-compared targets in one dispatch with a genuinely-changed target at
    // j >= 1 -- the only shape that can observe indexing the flags buffer by sizeof(uint32_t)
    // instead of by the descriptor-offset stride (#3031). Two persistent >=1 MiB writable buffers
    // bound to ONE dispatch, each already carrying a retained GPU-side result baseline from a prior
    // dispatch of the same item. The first (j=0) is re-dispatched with an unchanged fill value; the
    // second (j=1) is re-dispatched with a genuinely different one. A correct readback must publish
    // the new guest bytes for j=1; reading the padding between flags on a device whose storage-buffer
    // offset alignment exceeds four (lavapipe: 16) would silently report it unchanged instead,
    // exactly like j=0, and leave the stale baseline bytes in guest memory.
    {
        constexpr uint32_t compare_stride_target_bytes = 1u << 20;
        std::vector<uint8_t> compare_stride_hostA(compare_stride_target_bytes, 0);
        std::vector<uint8_t> compare_stride_hostB(compare_stride_target_bytes, 0);

        // Hand-written RDNA2: v4 = 0 (single invocation, no thread-index dependency needed); store
        // s4-s7 through the V# at s0-s3 (binding 0, j=0), then s12-s15 through the V# at s8-s11
        // (binding 1, j=1). Both stores are buffer_store_dwordx4, idxen, offset 0 -- word-for-word
        // the Dead Cells fill kernel's decoded MUBUF encoding above (opcode 7, VDATA/VADDR/SRSRC/
        // SOFFSET per rdna2_decode.cpp's own field layout), just re-targeted to a second V#/value
        // pair for the second store. VOP1 v_mov_b32 = 0x7E000000 | vdst<<17 | 1<<9 | src0, verified
        // against both `v0 = s4` (this file) and `v5 = y = 0` (image_copy_2d below).
        static const uint32_t two_target_fill[] = {
            0x7e080280u,               // v4 = 0
            0x7e000204u,               // v0 = s4
            0x7e020205u,               // v1 = s5
            0x7e040206u,               // v2 = s6
            0x7e060207u,               // v3 = s7
            0xe01c2000u, 0x80000004u,  // buffer_store_dwordx4 v[0:3], v4, s[0:3], offset0 idxen (j=0)
            0x7e0a020cu,               // v5 = s12
            0x7e0c020du,               // v6 = s13
            0x7e0e020eu,               // v7 = s14
            0x7e10020fu,               // v8 = s15
            0xe01c2000u, 0x80020504u,  // buffer_store_dwordx4 v[5:8], v4, s[8:11], offset0 idxen (j=1)
            0xbf810000u,               // s_endpgm
        };

        ShaderResourceTable compare_stride_rt;
        ShaderResource compare_stride_a;
        compare_stride_a.cls = ResourceClass::ConstantBuffer;
        compare_stride_a.format = DataFormat::Uint32;
        compare_stride_a.num_components = 4;
        compare_stride_a.binding = 0;
        compare_stride_a.sgpr_base = 0;
        compare_stride_a.stride = 4 * sizeof(uint32_t);
        compare_stride_a.gpu_addr = reinterpret_cast<uint64_t>(compare_stride_hostA.data());
        compare_stride_a.size = compare_stride_target_bytes;
        compare_stride_rt.resources.push_back(compare_stride_a);
        ShaderResource compare_stride_b;
        compare_stride_b.cls = ResourceClass::ConstantBuffer;
        compare_stride_b.format = DataFormat::Uint32;
        compare_stride_b.num_components = 4;
        compare_stride_b.binding = 1;
        compare_stride_b.sgpr_base = 8;
        compare_stride_b.stride = 4 * sizeof(uint32_t);
        compare_stride_b.gpu_addr = reinterpret_cast<uint64_t>(compare_stride_hostB.data());
        compare_stride_b.size = compare_stride_target_bytes;
        compare_stride_rt.resources.push_back(compare_stride_b);

        ComputeShaderConfig compare_stride_config;
        compare_stride_config.local_x = 1;
        compare_stride_config.user_sgprs = {
            0, 0, 0, 0,                                          // s0-3: A's V# (injected)
            0xa1a1a1a1u, 0xa2a2a2a2u, 0xa3a3a3a3u, 0xa4a4a4a4u,   // s4-7: A's baseline fill
            0, 0, 0, 0,                                          // s8-11: B's V# (injected)
            0xb1b1b1b1u, 0xb2b2b2b2u, 0xb3b3b3b3u, 0xb4b4b4b4u,   // s12-15: B's baseline fill
        };
        std::vector<uint32_t> compare_stride_spirv = recompile_compute(
            two_target_fill, std::size(two_target_fill), &compare_stride_rt, compare_stride_config);
        CHECK(!compare_stride_spirv.empty(), "#3032 two-target compare-flag kernel recompiles");
        if (!compare_stride_spirv.empty()) {
            auto compare_stride_report = validate_spirv_descriptor_interface(
                compare_stride_spirv, &compare_stride_rt, 0, SpirvShaderStage::Compute);
            CHECK(compare_stride_report.ok(),
                  "#3032 two-target compare-flag kernel's descriptor interface validates");

            ComputeItem compare_stride_item;
            compare_stride_item.spirv = compare_stride_spirv;
            compare_stride_item.resources =
                std::make_shared<ShaderResourceTable>(compare_stride_rt);
            compare_stride_item.launch.threads_x = compare_stride_item.launch.threads_y =
                compare_stride_item.launch.threads_z = 1;
            compare_stride_item.launch.local_x = compare_stride_item.launch.local_y =
                compare_stride_item.launch.local_z = 1;
            compare_stride_item.launch.groups_x = compare_stride_item.launch.groups_y =
                compare_stride_item.launch.groups_z = 1;
            compare_stride_item.user_sgprs = compare_stride_config.user_sgprs;
            compare_stride_item.code_addr = 0x3032000000ull;

            CHECK(prosper::frontend::execute_live_compute_items({compare_stride_item}),
                  "#3032 two-target dispatch establishes both retained GPU baselines");

            auto load_u32 = [](const std::vector<uint8_t>& bytes, size_t offset) {
                uint32_t value = 0;
                std::memcpy(&value, bytes.data() + offset, sizeof(value));
                return value;
            };
            const bool baseline_a_ok = load_u32(compare_stride_hostA, 0) == 0xa1a1a1a1u &&
                load_u32(compare_stride_hostA, 4) == 0xa2a2a2a2u &&
                load_u32(compare_stride_hostA, 8) == 0xa3a3a3a3u &&
                load_u32(compare_stride_hostA, 12) == 0xa4a4a4a4u;
            const bool baseline_b_ok = load_u32(compare_stride_hostB, 0) == 0xb1b1b1b1u &&
                load_u32(compare_stride_hostB, 4) == 0xb2b2b2b2u &&
                load_u32(compare_stride_hostB, 8) == 0xb3b3b3b3u &&
                load_u32(compare_stride_hostB, 12) == 0xb4b4b4b4u;
            CHECK(baseline_a_ok && baseline_b_ok,
                  "#3032 two-target dispatch publishes both targets' initial guest bytes");

            // Re-dispatch unchanged: nothing wrote either host buffer since the first dispatch, so
            // both A and B are eligible for the retained-baseline GPU comparator this time (j=0 and
            // j=1 respectively). Give the SECOND target (j=1) a genuinely different fill value; the
            // first (j=0) stays identical to prove the fast path still engages at all -- otherwise a
            // suite change that stopped exercising the comparator entirely would pass this arm too.
            ComputeItem compare_stride_item2 = compare_stride_item;
            compare_stride_item2.user_sgprs[12] = 0xc1c1c1c1u;
            compare_stride_item2.user_sgprs[13] = 0xc2c2c2c2u;
            compare_stride_item2.user_sgprs[14] = 0xc3c3c3c3u;
            compare_stride_item2.user_sgprs[15] = 0xc4c4c4c4u;

            const uint64_t compare_stride_skips_before =
                prosper::frontend::live_compute_buffer_gpu_result_skips();
            CHECK(prosper::frontend::execute_live_compute_items({compare_stride_item2}),
                  "#3032 two-target dispatch re-runs with only the SECOND target changed");
            CHECK(prosper::frontend::live_compute_buffer_gpu_result_skips() >
                      compare_stride_skips_before,
                  "#3032 unchanged first target still takes the retained GPU-comparator fast path");

            const bool target_a_unchanged =
                load_u32(compare_stride_hostA, 0) == 0xa1a1a1a1u &&
                load_u32(compare_stride_hostA, 4) == 0xa2a2a2a2u &&
                load_u32(compare_stride_hostA, 8) == 0xa3a3a3a3u &&
                load_u32(compare_stride_hostA, 12) == 0xa4a4a4a4u;
            CHECK(target_a_unchanged,
                  "#3032 GPU-identical first target (j=0) preserves its guest bytes");

            // The discriminating assertion. A wrong readback that indexes the flags buffer by
            // sizeof(uint32_t) reads j=1's flag from the padding between j=0's and j=1's dwords on
            // any device with an offset alignment above four (lavapipe: 16). That padding is zero,
            // so it reports j=1 as unchanged and the writeback loop takes the byte-preserving skip
            // path instead of copying the genuinely new GPU result into guest memory -- the same
            // silent-stale-read #3031 fixed for the descriptor offset, previously unguarded for the
            // readback.
            const bool target_b_updated =
                load_u32(compare_stride_hostB, 0) == 0xc1c1c1c1u &&
                load_u32(compare_stride_hostB, 4) == 0xc2c2c2c2u &&
                load_u32(compare_stride_hostB, 8) == 0xc3c3c3c3u &&
                load_u32(compare_stride_hostB, 12) == 0xc4c4c4c4u;
            CHECK(target_b_updated,
                  "#3032 genuinely-changed second target (j=1) publishes its new guest bytes, not "
                  "stale padding-read baseline bytes");
        }
    }

    std::fill(result.begin(), result.end(), 0xeeeeeeee);
    item.user_sgprs = alternate_config.user_sgprs;
    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "cached compute pipeline executes with updated user SGPR push constants");
    const uint32_t alternate_expected[4] = {
        alternate_config.user_sgprs[4], alternate_config.user_sgprs[5],
        alternate_config.user_sgprs[6], alternate_config.user_sgprs[7],
    };
    bool alternate_filled = true;
    for (uint32_t record = 0; record < records && alternate_filled; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            alternate_filled &= result[record * 4 + component] == alternate_expected[component];
    CHECK(alternate_filled,
          "cached compute pipeline observes per-dispatch user SGPR values");
    item.user_sgprs = config.user_sgprs;

    std::vector<uint32_t> replay_owned(launched_records * 4, 0xdddddddd);
    ShaderResource replay_buffer = buffer;
    replay_buffer.gpu_addr = 1; // Deliberately unreadable: replay must never dereference this identity.
    replay_buffer.host_data = reinterpret_cast<uint8_t*>(replay_owned.data());
    replay_buffer.host_data_size = replay_owned.size() * sizeof(uint32_t);
    item.resources = std::make_shared<ShaderResourceTable>();
    item.resources->resources.push_back(replay_buffer);
    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "production compute backend executes against replay-owned bytes");
    bool replay_filled = true;
    for (uint32_t record = 0; record < records && replay_filled; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            replay_filled &= replay_owned[record * 4 + component] == expected[component];
    CHECK(replay_filled, "compute writeback updates owned backing for a later replay operation");
    bool replay_padding_untouched = true;
    for (uint32_t record = records; record < launched_records; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            replay_padding_untouched &= replay_owned[record * 4 + component] == 0xdddddddd;
    CHECK(replay_padding_untouched, "captured/replay-owned backing preserves padded invocations");

    // Terminator 2D's startup uses device-global GDS. Capture its initial host-backed state, then
    // execute two ordered replay dispatches against the same materialized 64 KiB instance. The
    // first store wraps its byte address to 16 bits; the second narrows EXEC to lane zero. If the
    // store ignores EXEC, lanes 1..3 overwrite the first dispatch's dword at wrapped address 4.
    static const uint32_t gds_wrap_store[] = {
        0x7e0002ffu, 0x00010004u,  // v0 = byte address 0x10004 -> GDS byte address 4
        0x7e0202ffu, 0xdeadbeefu,  // v1 = value
        0xd8360000u, 0x00000100u,  // ds_write_b32 v0, v1 gds
        0xbf810000u,
    };
    static const uint32_t gds_exec_store[] = {
        0x34000082u,               // v_lshlrev_b32 v0, 2, v0 (one dword per lane)
        0x7e0202ffu, 0xcafebabeu,  // v1 = value
        0x7da40080u,               // v_cmpx_eq_u32 0, v0
        0xd8360000u, 0x00000100u,  // only lane zero stores at byte address 0
        0xbf810000u,
    };
    std::array<uint8_t, 64 * 1024> gds_initial{};
    ShaderResource gds_resource;
    gds_resource.cls = ResourceClass::ConstantBuffer;
    gds_resource.format = DataFormat::Uint32;
    gds_resource.num_components = 1;
    gds_resource.binding = kComputeInternalGdsBinding;
    gds_resource.size = gds_initial.size();
    gds_resource.stride = 4;
    gds_resource.host_data = gds_initial.data();
    gds_resource.host_data_size = gds_initial.size();
    auto gds_table = std::make_shared<ShaderResourceTable>();
    gds_table->resources.push_back(gds_resource);
    ComputeShaderConfig gds_config;
    auto make_gds_item = [&](const uint32_t* code_words, size_t word_count,
                             uint32_t threads, uint64_t order) {
        ComputeItem gds_item;
        gds_item.spirv = recompile_compute(code_words, word_count, gds_table.get(), gds_config);
        gds_item.resources = gds_table;
        gds_item.launch.threads_x = threads;
        gds_item.launch.local_x = threads;
        gds_item.launch.local_y = gds_item.launch.local_z = 1;
        gds_item.launch.groups_x = gds_item.launch.groups_y = gds_item.launch.groups_z = 1;
        gds_item.dispatch_index = order;
        gds_item.command_order = order;
        return gds_item;
    };
    ComputeItem gds_first = make_gds_item(gds_wrap_store, std::size(gds_wrap_store), 1, 10);
    ComputeItem gds_second = make_gds_item(gds_exec_store, std::size(gds_exec_store), 4, 20);
    CHECK(!gds_first.spirv.empty() && !gds_second.spirv.empty(),
          "compute GDS wrap and EXEC kernels recompile");
    GpuCaptureFile gds_capture;
    GpuCaptureMetadata gds_metadata;
    std::string gds_error;
    const std::vector<SubmitOperation> gds_operations = {
        {SubmitOperationKind::Dispatch, 10, 10},
        {SubmitOperationKind::Dispatch, 20, 20},
    };
    // Deferred runtime capture finishes after the backend mutates this shared table. Its replay
    // input must nevertheless be the pre-submit GDS bytes, paired with the exact realized compute
    // list obtained after execution.
    constexpr uint32_t kGdsPreSubmitSentinel = 0x10293847u;
    constexpr uint32_t kGdsPostSubmitSentinel = 0xfedcba98u;
    std::memcpy(gds_initial.data(), &kGdsPreSubmitSentinel,
                sizeof(kGdsPreSubmitSentinel));
    // Per-process path: this binary runs as two concurrent ctest cases, so a fixed name made both
    // read back the other run's bytes (#1613).
    const std::string deferred_gds_capture_path =
        prosper_test::test_scratch_file("prosper_deferred_gds_capture.prgcap");
    auto deferred_gds_pending = std::make_unique<PendingGpuCapture>();
    deferred_gds_pending->materialized = false;
    deferred_gds_pending->path = deferred_gds_capture_path;
    snapshot_pending_gpu_capture_compute_gds(
        deferred_gds_pending.get(), gds_initial.data(), gds_initial.size());
    std::memcpy(gds_initial.data(), &kGdsPostSubmitSentinel,
                sizeof(kGdsPostSubmitSentinel));
    const std::vector<DrawItem> deferred_gds_draws;
    const std::vector<ComputeItem> deferred_gds_computes = {gds_first, gds_second};
    CHECK(finish_requested_gpu_capture(
              std::move(deferred_gds_pending), {}, gds_error, &deferred_gds_draws,
              &deferred_gds_computes, &gds_operations),
          "deferred capture writes exact compute operations with snapshotted GDS input");
    GpuCaptureFile deferred_gds_capture;
    GpuReplayFrame deferred_gds_replay;
    const bool deferred_gds_reopened = read_gpu_capture(
        deferred_gds_capture_path, deferred_gds_capture, gds_error) &&
        materialize_gpu_replay(deferred_gds_capture, deferred_gds_replay, gds_error);
    const uint32_t* deferred_gds_words = deferred_gds_reopened &&
        !deferred_gds_replay.computes.empty() &&
        deferred_gds_replay.computes[0].resources &&
        !deferred_gds_replay.computes[0].resources->resources.empty()
            ? reinterpret_cast<const uint32_t*>(
                  deferred_gds_replay.computes[0].resources->resources[0].host_data)
            : nullptr;
    CHECK(deferred_gds_words && deferred_gds_words[0] == kGdsPreSubmitSentinel,
          "deferred capture replay restores pre-submit persistent GDS, not post-submit bytes");
    std::remove(deferred_gds_capture_path.c_str());
    gds_initial.fill(0);
    CHECK(capture_submit_items({}, {gds_first, gds_second}, gds_operations, gds_metadata,
                               [](uint64_t, uint8_t*, size_t) { return size_t{0}; },
                               gds_capture, gds_error),
          "capture records host-backed compute GDS without reading guest address zero");
    std::vector<uint8_t> gds_capture_bytes;
    GpuCaptureFile gds_loaded;
    GpuReplayFrame gds_replay;
    CHECK(serialize_gpu_capture(gds_capture, gds_capture_bytes, gds_error) &&
          deserialize_gpu_capture(gds_capture_bytes, gds_loaded, gds_error) &&
          materialize_gpu_replay(gds_loaded, gds_replay, gds_error) &&
          gds_replay.computes.size() == 2 &&
          gds_replay.computes[0].resources->resources[0].host_data ==
              gds_replay.computes[1].resources->resources[0].host_data,
          "capture v22 materializes one persistent GDS instance across ordered dispatches");
    uint32_t gds_notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t, uint64_t, const char*) { ++gds_notifications; });
    CHECK(prosper::frontend::execute_live_compute_items(gds_replay.computes),
          "production live backend executes captured GDS stores");
    set_guest_gpu_write_observer({});
    const auto* gds_words = reinterpret_cast<const uint32_t*>(
        gds_replay.computes[0].resources->resources[0].host_data);
    CHECK(gds_words && gds_words[0] == 0xcafebabeu && gds_words[1] == 0xdeadbeefu &&
              gds_words[2] == 0 && gds_words[3] == 0,
          "GDS stores honor 16-bit address wrap, EXEC predication, and cross-dispatch persistence");
    CHECK(gds_notifications == 0,
          "GPU-internal GDS writeback does not invalidate guest address zero");

    // Astro Bot's four indirect-list counters prove the GDS M0 layout independently of the LDS
    // append tests: M0[31:16] is the byte base and M0[15:0] is the allocation size. The exact live
    // value 0x0c600020 plus offset 0x10 must append at 0xc70, not at the old low-half address 0x30.
    // A second kernel crosses the 64 KiB boundary, proving that base+offset wraps in the GDS domain.
    static const uint32_t gds_m0_high_append[] = {
        0xbefc03ffu, 0x0c600020u,  // s_mov_b32 m0, {base=0xc60,size=0x20}
        0xd8fa0010u, 0x00000000u,  // ds_append v0 offset:0x10 gds -> 0xc70
        0xbf810000u,
    };
    static const uint32_t gds_m0_wrap_append[] = {
        0xbefc03ffu, 0xfff00020u,  // s_mov_b32 m0, {base=0xfff0,size=0x20}
        0xd8fa0030u, 0x00000000u,  // ds_append v0 offset:0x30 gds -> 0x20 after wrap
        0xbf810000u,
    };
    gds_initial.fill(0);
    const auto* gds_m0_words = reinterpret_cast<const uint32_t*>(gds_initial.data());
    CHECK(gds_m0_words[0] == 0 && gds_m0_words[0x20 / 4] == 0 &&
              gds_m0_words[0x30 / 4] == 0 && gds_m0_words[0x50 / 4] == 0 &&
              gds_m0_words[0xc70 / 4] == 0,
          "GDS M0 fixture starts with zeroed authoritative host backing");
    // Match Astro's 1x1x1 producer. The earlier store fixture intentionally used the default
    // 64-lane workgroup; leaving that config in place here would append 64 active lanes while the
    // launch metadata claimed one, making the address assertion depend on an unrelated count.
    gds_config.local_x = 1;
    ComputeItem gds_high_append = make_gds_item(
        gds_m0_high_append, std::size(gds_m0_high_append), 1, 30);
    ComputeItem gds_wrap_append = make_gds_item(
        gds_m0_wrap_append, std::size(gds_m0_wrap_append), 1, 40);
    CHECK(!gds_high_append.spirv.empty() && !gds_wrap_append.spirv.empty(),
          "GDS append kernels with nonzero M0 base recompile");
    CHECK(prosper::frontend::execute_live_compute_items({gds_high_append, gds_wrap_append}),
          "production compute backend executes GDS M0 base and wrap kernels");
    printf("  GDS M0 words [0]=%u [0x20]=%u [0x30]=%u [0x50]=%u [0xc70]=%u\n",
           gds_m0_words[0], gds_m0_words[0x20 / 4], gds_m0_words[0x30 / 4],
           gds_m0_words[0x50 / 4], gds_m0_words[0xc70 / 4]);
    CHECK(gds_m0_words[0xc70 / 4] == 1 && gds_m0_words[0x20 / 4] == 1 &&
              gds_m0_words[0x30 / 4] == 0 && gds_m0_words[0x50 / 4] == 0,
          "GDS append uses M0 high-half base, wraps at 64 KiB, and rejects low-half aliases");

    // --- #590: the live backend's storage-IMAGE path. The same 1D image-copy kernel that
    // test_storage_image_copy proves against the raw harness, executed through the PRODUCTION
    // backend with Unorm8x4 guest-style backing — exercising the full chain: channel unpack
    // (bytes -> float-bit uvec4 texels), the R32G32B32A32_UINT image contract, and the pack-back
    // writeback (float bits -> clamped bytes). Unorm8 pack(unpack(b)) == b exactly, so a bit-exact
    // dst==src is the correctness assertion.
    static const uint32_t image_copy[] = {
        0x7E080300u, 0xF0000F00u, 0x00000004u, 0xBF8C3F70u, 0xF0200F00u, 0x00020004u, 0xBF810000u,
    };
    static const uint32_t image_copy_2d[] = {
        0x7E080300u,             // v4 = x from the shell input
        0x7E0A0280u,             // v5 = y = 0
        0xF0000F08u, 0x00000004u, 0xBF8C3F70u,
        0xF0200F08u, 0x00020004u, 0xBF810000u,
    };
    static const uint32_t image_copy_3d[] = {
        0x7E080300u,             // v4 = x from the shell input
        0x7E0A0280u,             // v5 = y = 0
        0x7E0C0280u,             // v6 = z = 0
        // MIMG.DIM lives in bits [5:3] (rdna2_decode.cpp: `(w >> 3) & 0x7`), so DIM=3D is 0x10 —
        // 0x08 is DIM=2D, exactly what image_copy_2d above encodes. This fixture prepares a third
        // address register and calls itself 3D, so the DIM field has to agree: with 0x08 the
        // recompiler faithfully emitted a 2D SPIR-V image addressed by two coordinates while the
        // backend bound the 3D view its 3D resource descriptor calls for, and a 3D view bound to a
        // descriptor whose shader declares Dim 2D is undefined in Vulkan (#1690).
        0xF0000F10u, 0x00000004u, 0xBF8C3F70u,
        0xF0200F10u, 0x00020004u, 0xBF810000u,
    };
    static const uint32_t image_copy_2d_array[] = {
        0x7E080300u,             // v4 = x from the shell input
        0x7E0A0280u,             // v5 = y = 0
        0x7E0C0282u,             // v6 = array layer 2
        0xF0000F28u, 0x00000004u, 0xBF8C3F70u,
        0xF0200F28u, 0x00020004u, 0xBF810000u,
    };
    const uint32_t W = 64;
    std::vector<uint32_t> lane_index(W);
    for (uint32_t i = 0; i < W; i++) lane_index[i] = i;      // shell input: v0 = input[gid] = gid
    std::vector<uint32_t> dummy(4, 0);
    std::vector<uint8_t> img_src(W * 4), img_dst(W * 4, 0xEE);
    for (uint32_t i = 0; i < W * 4; i++) img_src[i] = (uint8_t)(i * 37 + 5);
    ShaderResourceTable irt;
    auto add_buffer = [&](uint32_t binding, void* data, uint32_t size) {
        ShaderResource b{};
        b.cls = ResourceClass::ConstantBuffer;
        b.binding = binding;
        b.gpu_addr = (uint64_t)(uintptr_t)data;
        b.size = size;
        irt.resources.push_back(b);
    };
    add_buffer(0, lane_index.data(), W * sizeof(uint32_t));
    add_buffer(1, dummy.data(), 16);
    add_buffer(2, dummy.data(), 16);
    add_buffer(3, dummy.data(), 16);
    auto add_image = [&](uint32_t binding, uint32_t sgpr, void* data, uint32_t size) {
        ShaderResource im{};
        im.cls = ResourceClass::StorageImage;
        im.img_dim = 0;                     // 1D
        im.binding = binding;
        im.sgpr_base = sgpr;
        im.format = DataFormat::Unorm8;
        im.num_components = 4;
        im.width = W; im.height = 1;
        im.gpu_addr = (uint64_t)(uintptr_t)data;
        im.size = size;
        irt.resources.push_back(im);
    };
    add_image(4, 0, img_src.data(), W * 4);
    add_image(5, 8, img_dst.data(), W * 4);
    std::vector<uint32_t> image_spirv = recompile_valu(
        image_copy, sizeof(image_copy) / sizeof(image_copy[0]), 1, 0, &irt);
    CHECK(!image_spirv.empty(), "storage-image copy kernel recompiles against the game-style table");
    if (!image_spirv.empty()) {
        ComputeItem image_item;
        image_item.spirv = image_spirv;
        image_item.resources = std::make_shared<ShaderResourceTable>(irt);
        image_item.launch.threads_x = W;
        image_item.launch.local_x = 64;
        image_item.launch.groups_x = 1;
        image_item.launch.local_y = image_item.launch.local_z = 1;
        image_item.launch.groups_y = image_item.launch.groups_z = 1;
        image_item.code_addr = 0x590590;
        CHECK(prosper::frontend::execute_live_compute_items({image_item}),
              "live backend executes the storage-image copy dispatch (#590)");
        uint32_t bad = 0;
        for (uint32_t i = 0; i < W * 4; i++) bad += img_dst[i] != img_src[i];
        if (bad) std::printf("  image copy mismatched bytes = %u/%u (b0 src=%02x dst=%02x)\n",
                             bad, W * 4, img_src[0], img_dst[0]);
        CHECK(bad == 0, "Unorm8 unpack -> uvec4 copy -> pack writeback is byte-exact (#590)");
    }

    // GTA V's heavy Windows transition copies two 2560x1440 single-channel surfaces and writes a
    // 3840x2160 Uint8x4 target. Keeping them in the raw RGBA32_UINT interchange image inflated the
    // allocations to 59/127 MiB; exercise the device-gated typed path end to end, including the
    // exact reflected image format.
    auto run_native_typed_copy = [&](DataFormat format, uint32_t spirv_format,
                                     SpirvImageNumericClass numeric_class,
                                     uint32_t components, uint32_t texel_bytes,
                                     const std::vector<uint8_t>& source,
                                     DataFormat sampled_format = DataFormat::Unknown) {
        std::vector<uint8_t> destination(source.size(), 0xa5);
        ShaderResourceTable table;
        auto add_uint_image = [&](uint32_t binding, uint32_t sgpr, void* data) {
            ShaderResource image{};
            image.cls = ResourceClass::StorageImage;
            image.img_dim = 1;
            image.binding = binding;
            image.sgpr_base = sgpr;
            image.format = format;
            image.num_components = components;
            image.width = W;
            image.height = image.depth = 1;
            image.gpu_addr = reinterpret_cast<uint64_t>(data);
            image.size = W * texel_bytes;
            table.resources.push_back(image);
        };
        add_uint_image(4, 0, const_cast<uint8_t*>(source.data()));
        add_uint_image(5, 8, destination.data());
        ComputeShaderConfig config;
        config.user_sgprs.resize(16);
        config.local_x = W;
        config.local_y = config.local_z = 1;
        config.tidig_comp_cnt = 0;
        config.native_storage_format_support =
            native_storage_format_support_bit(format, components);
        const std::vector<uint32_t> spirv = recompile_compute(
            image_copy_2d, std::size(image_copy_2d), &table, config);
        const DescriptorValidationReport report = validate_spirv_descriptor_interface(
            spirv, &table, 0, SpirvShaderStage::Compute, false);
        const bool typed = !spirv.empty() && report.ok() && report.descriptors.size() == 2 &&
            std::all_of(report.descriptors.begin(), report.descriptors.end(),
                        [&](const SpirvDescriptorBinding& descriptor) {
                            return descriptor.kind == SpirvDescriptorKind::StorageImage &&
                                   descriptor.image_numeric_class == numeric_class &&
                                   descriptor.storage_image_format == spirv_format;
                        });
        if (!typed) return std::tuple{false, destination, false, uint32_t{0}};
        ComputeItem item;
        item.spirv = spirv;
        item.resources = std::make_shared<ShaderResourceTable>(table);
        item.launch.threads_x = W;
        item.launch.local_x = W;
        item.launch.groups_x = 1;
        item.launch.threads_y = item.launch.threads_z = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = format == DataFormat::Float32 ? 0x59059fu
            : format == DataFormat::Uint32 ? 0x5905a0u
            : format == DataFormat::Uint16 ? 0x5905a1u : 0x5905a2u;
        // The first sight of a write-only program poison-proves full coverage. The following
        // ordered submit is the production state: the proven output may retain/export its exact
        // typed image to the immediately following graphics consumer.
        const bool coverage_proven =
            prosper::frontend::execute_live_compute_items({item});
        item.dispatch_index = format == DataFormat::Float32 ? 50u
            : format == DataFormat::Uint32 ? 51u
            : format == DataFormat::Uint16 ? 52u : 53u;
        item.command_order = 10;
        DrawItem consumer;
        consumer.draw_index = format == DataFormat::Float32 ? 60u
            : format == DataFormat::Uint32 ? 61u
            : format == DataFormat::Uint16 ? 62u : 63u;
        consumer.command_order = 20;
        bool imported = false;
        uint32_t imported_format = 0;
        const OrderedSubmitResult ordered = execute_ordered_items(
            {{SubmitOperationKind::Dispatch, item.dispatch_index, item.command_order},
             {SubmitOperationKind::Draw, consumer.draw_index, consumer.command_order}},
            {consumer}, {item},
            [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                ShaderResource sampled = table.resources.back();
                sampled.cls = ResourceClass::Texture;
                if (sampled_format != DataFormat::Unknown)
                    sampled.format = sampled_format;
                prosper::frontend::LiveComputeImageImport compute_import;
                imported = prosper::frontend::import_live_compute_storage_image(
                    sampled, sampled.size, compute_import) && compute_import.valid();
                if (imported) imported_format = compute_import.native_format;
                return RenderedFrame{};
            },
            [&](const std::vector<ComputeItem>& items) {
                return prosper::frontend::execute_live_compute_items(items);
            },
            1, 1);
        return std::tuple{
            coverage_proven && ordered.compute_executed,
            destination, imported, imported_format};
    };
    {
        static const uint32_t fill_r32_2d[] = {
            0x7e080300u,                         // v4 = x
            0x7e0a0280u,                         // v5 = y = 0
            0x7e0002f2u,                         // v0 = 1.0f
            0xf0200108u, 0x00020004u,            // IMAGE_STORE x at (v4,v5)
            0xbf810000u,
        };
        std::vector<uint32_t> destination(W, 0xdeadbeefu);
        ShaderResource output{};
        output.cls = ResourceClass::StorageImage;
        output.img_dim = 1;
        output.binding = 5;
        output.sgpr_base = 8;
        output.format = DataFormat::Float32;
        output.num_components = 1;
        output.width = W;
        output.height = output.depth = 1;
        output.gpu_addr = reinterpret_cast<uint64_t>(destination.data());
        output.size = static_cast<uint32_t>(destination.size() * sizeof(uint32_t));
        ShaderResourceTable table;
        table.resources.push_back(output);
        ComputeShaderConfig config;
        config.user_sgprs.resize(16);
        config.local_x = W;
        config.local_y = config.local_z = 1;
        config.tidig_comp_cnt = 0;
        config.native_storage_format_support =
            native_storage_format_support_bit(DataFormat::Float32, 1);
        const std::vector<uint32_t> spirv = recompile_compute(
            fill_r32_2d, std::size(fill_r32_2d), &table, config);
        const DescriptorValidationReport report = validate_spirv_descriptor_interface(
            spirv, &table, 0, SpirvShaderStage::Compute, false);
        const SpirvDescriptorBinding* binding =
            find_spirv_descriptor_binding(report, 0, output.binding);
        // Native float storage keeps the SPIR-V image format Unknown and carries the exact
        // R32_SFLOAT choice in the resource/device contract. That preserves the existing
        // formatless float shader ABI; the imported VkFormat assertion below pins the concrete
        // device image separately.
        CHECK(!spirv.empty() && report.ok() && binding && binding->storage_float &&
                  binding->image_numeric_class == SpirvImageNumericClass::Float &&
                  binding->storage_image_format == 0u,
              "R32_SFLOAT fill reflects native float storage");
        if (!spirv.empty() && report.ok() && binding && binding->storage_float) {
            ComputeItem item;
            item.spirv = spirv;
            item.resources = std::make_shared<ShaderResourceTable>(table);
            item.launch.threads_x = W;
            item.launch.local_x = W;
            item.launch.groups_x = 1;
            item.launch.threads_y = item.launch.threads_z = 1;
            item.launch.local_y = item.launch.local_z = 1;
            item.launch.groups_y = item.launch.groups_z = 1;
            item.code_addr = 0x59059fu;
            const bool coverage_proven =
                prosper::frontend::execute_live_compute_items({item});
            item.dispatch_index = 50;
            item.command_order = 10;
            DrawItem consumer;
            consumer.draw_index = 60;
            consumer.command_order = 20;
            bool imported = false;
            // #3307. The borrow census is asserted on the REAL path -- a real device, a real
            // producer dispatch, a real ordered submit -- because the unit test can only prove the
            // classifier is self-consistent. The three snapshots bracket one hit and one
            // deliberately mis-keyed miss taken back to back inside the same consumer callback.
            const auto census_before = prosper::frontend::live_compute_image_borrow_census();
            prosper::frontend::ComputeImageBorrowCensusSnapshot census_after_hit{};
            prosper::frontend::ComputeImageBorrowCensusSnapshot census_after_miss{};
            const OrderedSubmitResult ordered = execute_ordered_items(
                {{SubmitOperationKind::Dispatch, item.dispatch_index, item.command_order},
                 {SubmitOperationKind::Draw, consumer.draw_index, consumer.command_order}},
                {consumer}, {item},
                [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                    ShaderResource sampled = output;
                    sampled.cls = ResourceClass::Texture;
                    prosper::frontend::LiveComputeImageImport compute_import;
                    imported = prosper::frontend::import_live_compute_storage_image(
                        sampled, sampled.size, compute_import) && compute_import.valid() &&
                        compute_import.native_format == 100u &&
                        compute_import.producer_command_order == item.command_order;
                    census_after_hit = prosper::frontend::live_compute_image_borrow_census();
                    // The same guest allocation with ONE key field changed. The borrow must miss,
                    // and the near-miss scan must name that field -- otherwise "nothing is cached
                    // under this key" and "the producer never ran" stay indistinguishable, which
                    // is exactly the ambiguity this census exists to remove.
                    ShaderResource mismatched = sampled;
                    mismatched.tile_mode = sampled.tile_mode + 1u;
                    prosper::frontend::LiveComputeImageImport mismatched_import;
                    (void)prosper::frontend::import_live_compute_storage_image(
                        mismatched, mismatched.size, mismatched_import);
                    census_after_miss = prosper::frontend::live_compute_image_borrow_census();
                    return RenderedFrame{};
                },
                [&](const std::vector<ComputeItem>& items) {
                    return prosper::frontend::execute_live_compute_items(items);
                },
                1, 1);
            CHECK(coverage_proven && ordered.compute_executed &&
                      std::all_of(destination.begin(), destination.end(),
                                  [](uint32_t value) { return value == 0x3f800000u; }),
                  "typed R32_SFLOAT fill executes and writes every finite value exactly");
            CHECK(imported,
                  "same-submit graphics consumer leases the exact typed R32_SFLOAT result");
            {
                using Outcome = prosper::frontend::ComputeImageBorrowOutcome;
                using Publish = prosper::frontend::ComputeImagePublishDecline;
                using Field = prosper::frontend::ComputeImageKeyField;
                constexpr size_t kHit = static_cast<size_t>(Outcome::Hit);
                constexpr size_t kNoEntry = static_cast<size_t>(Outcome::NoCacheEntry);
                constexpr size_t kAuthorized = static_cast<size_t>(Publish::Authorized);
                constexpr size_t kTileMode = static_cast<size_t>(Field::TileMode);
                constexpr size_t kWidth = static_cast<size_t>(Field::Width);
                constexpr size_t kFormat = static_cast<size_t>(Field::Format);
                constexpr size_t kVkFormat = static_cast<size_t>(Field::VkFormat);
                CHECK(census_after_hit.outcomes[kHit] == census_before.outcomes[kHit] + 1,
                      "the borrow census records the same-submit lease as a hit");
                CHECK(census_after_hit.publishes[kAuthorized] >
                          census_before.publishes[kAuthorized],
                      "the borrow census records the producer's export authorization");
                CHECK(census_after_miss.outcomes[kNoEntry] ==
                          census_after_hit.outcomes[kNoEntry] + 1,
                      "a one-field key difference is recorded as a cache-entry miss");
                // Float32x1 aliases to R32_UINT, so the importer retries under the producer
                // identity before giving up. The retry is counted, and its key is deliberately NOT
                // scanned.
                CHECK(census_after_miss.alias_retries == census_after_hit.alias_retries + 1,
                      "the format-alias retry is counted separately from the outcome");
                CHECK(census_after_miss.no_entry_same_addr ==
                          census_after_hit.no_entry_same_addr + 1 &&
                      census_after_miss.key_field_diffs[kTileMode] ==
                          census_after_hit.key_field_diffs[kTileMode] + 1,
                      "the near-miss scan finds the same-address entry and names tile_mode");
                // ...and names only the fields that actually differ. A mask that set every bit
                // would satisfy the arm above and say nothing.
                CHECK(census_after_miss.key_field_diffs[kWidth] ==
                          census_after_hit.key_field_diffs[kWidth],
                      "the near-miss scan does not attribute a field that matched");
                // The arm that pins WHICH key the mask describes, and the reason it exists: the
                // alias retry rewrites `format` and `vk_format`, so recording the retry's mask
                // instead of the exact key's would report those two as differing on a miss whose
                // only real difference is `tile_mode`. Only the exact key is scanned, so both must
                // be untouched. Without this pair the tile_mode/width arms above hold identically
                // for either key and say nothing about attribution.
                CHECK(census_after_miss.key_field_diffs[kFormat] ==
                          census_after_hit.key_field_diffs[kFormat] &&
                      census_after_miss.key_field_diffs[kVkFormat] ==
                          census_after_hit.key_field_diffs[kVkFormat],
                      "the near-miss mask is the exact key's, not the format-alias retry's");
                // A scan is recorded per scan PERFORMED. The retry missed too and was not scanned,
                // so exactly one scan is recorded against two NoCacheEntry outcomes' worth of
                // lookups -- the distinction the counter's name now carries.
                CHECK(census_after_miss.exact_key_scans == census_after_hit.exact_key_scans + 1,
                      "one scan is recorded per exact-key lookup, not per miss observed");
            }
            {
                // The classifier's eleven terms are proven equivalent to the chain they replaced by
                // a 2048-case sweep in `test_compute_image_borrow_census` -- but that sweep operates
                // on the abstract booleans. The hand-written `gates.* = <descriptor field>`
                // assignments that PRODUCE those booleans are where a lost `!` would actually live,
                // and nothing covered them. Drive them from the real path, one field at a time off a
                // descriptor that is otherwise known to classify cleanly.
                //
                // Ten of the eleven decline terms; `NoContext` is structurally undrivable here,
                // because reaching this point at all requires a live compute backend. Do not read
                // this array as covering all eleven.
                using Decline = prosper::frontend::ComputeImageImportDecline;
                ShaderResource base = output;
                base.cls = ResourceClass::Texture;
                uint8_t host_bytes[4] = {};
                struct WiringCase {
                    const char* what;
                    Decline want;
                    void (*mutate)(ShaderResource&, uint64_t&, uint8_t*);
                };
                static const WiringCase wiring[] = {
                    {"gpu_addr", Decline::NoGuestAddress,
                     [](ShaderResource& r, uint64_t&, uint8_t*) { r.gpu_addr = 0; }},
                    {"class", Decline::NotTextureClass,
                     [](ShaderResource& r, uint64_t&, uint8_t*) {
                         r.cls = ResourceClass::StorageImage; }},
                    {"host_data", Decline::HostBackedData,
                     [](ShaderResource& r, uint64_t&, uint8_t* host) { r.host_data = host; }},
                    {"guest_bytes", Decline::GuestByteRange,
                     [](ShaderResource&, uint64_t& bytes, uint8_t*) { bytes = 0; }},
                    {"guest_bytes_wide", Decline::GuestByteRange,
                     [](ShaderResource&, uint64_t& bytes, uint8_t*) {
                         bytes = uint64_t(UINT32_MAX) + 1; }},
                    {"shape", Decline::Shape,
                     [](ShaderResource& r, uint64_t&, uint8_t*) { r.img_dim = 4; }},
                    {"mip_levels", Decline::MipLevels,
                     [](ShaderResource& r, uint64_t&, uint8_t*) { r.declared_mip_levels = 2; }},
                    {"mip_tail", Decline::MipTail,
                     [](ShaderResource& r, uint64_t&, uint8_t*) { r.in_mip_tail = true; }},
                    {"srgb", Decline::Srgb,
                     [](ShaderResource& r, uint64_t&, uint8_t*) { r.srgb = true; }},
                    {"depth_compare", Decline::DepthCompare,
                     [](ShaderResource& r, uint64_t&, uint8_t*) { r.depth_compare = true; }},
                    {"native_format", Decline::NativeFormat,
                     [](ShaderResource& r, uint64_t&, uint8_t*) {
                         r.format = DataFormat::Unorm8; r.num_components = 3; }},
                };
                // POSITIVE CONTROL, and it is load-bearing rather than decorative. The classifier
                // returns the FIRST failing term, so a `base` that already declined would let every
                // case below pass while testing nothing -- each would report its own term only
                // because the mutation happens to precede the pre-existing failure, or would report
                // the wrong term entirely. Pin that the unmutated descriptor is accepted, so a
                // future edit to `output` cannot quietly hollow out the whole array.
                {
                    const auto before = prosper::frontend::live_compute_image_borrow_census();
                    prosper::frontend::LiveComputeImageImport control_import;
                    (void)prosper::frontend::import_live_compute_storage_image(
                        base, base.size, control_import);
                    const auto after = prosper::frontend::live_compute_image_borrow_census();
                    const size_t accepted_bucket = static_cast<size_t>(Decline::None);
                    CHECK(after.declines[accepted_bucket] == before.declines[accepted_bucket] + 1,
                          "the unmutated wiring probe is ACCEPTED, so each mutated arm below is "
                          "testing its own term rather than a pre-existing decline");
                }
                for (const WiringCase& c : wiring) {
                    ShaderResource probe = base;
                    uint64_t bytes = probe.size;
                    c.mutate(probe, bytes, host_bytes);
                    const auto before = prosper::frontend::live_compute_image_borrow_census();
                    prosper::frontend::LiveComputeImageImport unused_import;
                    const bool imported_probe =
                        prosper::frontend::import_live_compute_storage_image(
                            probe, bytes, unused_import);
                    const auto after = prosper::frontend::live_compute_image_borrow_census();
                    const size_t want = static_cast<size_t>(c.want);
                    const size_t accepted_bucket = static_cast<size_t>(Decline::None);
                    if (imported_probe || after.declines[want] != before.declines[want] + 1 ||
                        after.declines[accepted_bucket] != before.declines[accepted_bucket]) {
                        std::printf("FAIL: import gate wiring for %s\n", c.what);
                        ++fails;
                    }
                    ++checks;
                }
            }
        }
    }
    {
        std::vector<uint8_t> source(W * sizeof(uint32_t));
        for (uint32_t x = 0; x < W; ++x) {
            const uint32_t value = 0xf0120000u ^ (x * 0x1020304u);
            std::memcpy(source.data() + x * sizeof(value), &value, sizeof(value));
        }
        const auto [executed, destination, imported, imported_format] = run_native_typed_copy(
            DataFormat::Uint32, kSpirvImageFormatR32ui, SpirvImageNumericClass::Uint,
            1, sizeof(uint32_t), source);
        CHECK(executed && destination == source,
              "typed R32_UINT storage copy reflects and executes byte-exactly");
        CHECK(imported && imported_format == 98u, // VK_FORMAT_R32_UINT
              "same-submit graphics consumer leases the exact typed R32_UINT result");
    }
    {
        std::vector<uint8_t> source(W * sizeof(uint16_t));
        for (uint32_t x = 0; x < W; ++x) {
            const uint16_t value = static_cast<uint16_t>(0xf000u ^ (x * 0x123u));
            std::memcpy(source.data() + x * sizeof(value), &value, sizeof(value));
        }
        const auto [executed, destination, imported, imported_format] = run_native_typed_copy(
            DataFormat::Uint16, kSpirvImageFormatR16ui, SpirvImageNumericClass::Uint,
            1, sizeof(uint16_t), source);
        CHECK(executed && destination == source,
              "typed R16_UINT storage copy reflects and executes byte-exactly");
        CHECK(imported && imported_format == 74u, // VK_FORMAT_R16_UINT
              "same-submit graphics consumer leases the exact typed R16_UINT result");
        const auto [alias_executed, alias_destination, alias_imported, alias_format] =
            run_native_typed_copy(DataFormat::Uint16, kSpirvImageFormatR16ui,
                                  SpirvImageNumericClass::Uint, 1,
                                  sizeof(uint16_t), source, DataFormat::Unorm16);
        CHECK(alias_executed && alias_destination == source,
              "typed R16_UINT storage remains byte-exact for an R16_UNORM graphics consumer");
        CHECK(alias_imported && alias_format == 70u, // VK_FORMAT_R16_UNORM
              "same-submit R16_UNORM consumer leases a mutable view of the exact R16_UINT result");
        const auto [fp16_executed, fp16_destination, fp16_imported, fp16_format] =
            run_native_typed_copy(DataFormat::Uint16, kSpirvImageFormatR16ui,
                                  SpirvImageNumericClass::Uint, 1,
                                  sizeof(uint16_t), source, DataFormat::Float16);
        CHECK(fp16_executed && fp16_destination == source,
              "typed R16_UINT storage remains byte-exact for an R16_SFLOAT graphics consumer");
        CHECK(fp16_imported && fp16_format == 76u, // VK_FORMAT_R16_SFLOAT
              "same-submit R16_SFLOAT consumer leases a mutable view of the exact R16_UINT result");
    }
    {
        // GTA V writes a six-layer R16_UINT DIM=2D_ARRAY shadow allocation, then samples the same
        // bytes through a DIM=CUBE descriptor. The graphics recompiler's established cube ABI is a
        // vertically stacked 2D image, so import must preserve the exact producer identity while
        // explicitly requesting a six-layer device-local stack copy.
        constexpr uint32_t layers = 6;
        static const uint32_t fill_r16_2d_array[] = {
            0x7e080300u,                         // v4 = x
            0x7e0a0301u,                         // v5 = y
            0x7e0c0302u,                         // v6 = array layer
            0x7e000302u,                         // v0 = layer value
            0xf0200128u, 0x00020004u,            // IMAGE_STORE x at (v4,v5,v6), DIM=2D_ARRAY
            0xbf810000u,
        };
        const size_t layer_bytes = W * sizeof(uint16_t);
        std::vector<uint8_t> destination(layer_bytes * layers, 0xa5);
        ShaderResource output{};
        output.cls = ResourceClass::StorageImage;
        output.img_dim = 5;
        output.binding = 5;
        output.sgpr_base = 8;
        output.format = DataFormat::Uint16;
        output.num_components = 1;
        output.width = W;
        output.height = 1;
        output.depth = layers;
        output.gpu_addr = reinterpret_cast<uint64_t>(destination.data());
        output.size = static_cast<uint32_t>(destination.size());
        output.linear_row_pitch_bytes = static_cast<uint32_t>(layer_bytes);
        output.layer_stride_bytes = static_cast<uint32_t>(layer_bytes);
        ShaderResourceTable table;
        table.resources.push_back(output);
        ComputeShaderConfig config;
        config.user_sgprs.resize(16);
        config.local_x = W;
        config.local_y = 1;
        config.local_z = layers;
        config.tidig_comp_cnt = 2;
        config.native_storage_format_support =
            native_storage_format_support_bit(DataFormat::Uint16, 1);
        const std::vector<uint32_t> spirv = recompile_compute(
            fill_r16_2d_array, std::size(fill_r16_2d_array), &table, config);
        const DescriptorValidationReport report = validate_spirv_descriptor_interface(
            spirv, &table, 0, SpirvShaderStage::Compute, false);
        const SpirvDescriptorBinding* binding =
            find_spirv_descriptor_binding(report, 0, output.binding);
        CHECK(!spirv.empty() && report.ok() && binding && binding->image_arrayed &&
                  binding->image_numeric_class == SpirvImageNumericClass::Uint &&
                  binding->storage_image_format == kSpirvImageFormatR16ui,
              "six-layer R16 producer reflects exact arrayed typed storage");
        if (!spirv.empty() && report.ok() && binding) {
            ComputeItem item;
            item.spirv = spirv;
            item.resources = std::make_shared<ShaderResourceTable>(table);
            item.launch.threads_x = W;
            item.launch.threads_y = 1;
            item.launch.threads_z = layers;
            item.launch.local_x = W;
            item.launch.local_y = 1;
            item.launch.local_z = layers;
            item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
            item.code_addr = 0x5905a3u;
            const bool coverage_proven = prosper::frontend::execute_live_compute_items({item});
            item.dispatch_index = 54;
            item.command_order = 10;
            DrawItem consumer;
            consumer.draw_index = 64;
            consumer.command_order = 20;
            bool imported = false;
            bool normalized_imported = false;
            bool mismatches_rejected = false;
            const OrderedSubmitResult ordered = execute_ordered_items(
                {{SubmitOperationKind::Dispatch, item.dispatch_index, item.command_order},
                 {SubmitOperationKind::Draw, consumer.draw_index, consumer.command_order}},
                {consumer}, {item},
                [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                    ShaderResource cube = output;
                    cube.cls = ResourceClass::Texture;
                    cube.img_dim = 3;
                    prosper::frontend::LiveComputeImageImport compute_import;
                    imported = prosper::frontend::import_live_compute_storage_image(
                        cube, cube.size, compute_import) && compute_import.valid() &&
                        compute_import.native_format == 74u &&
                        compute_import.vertical_stack_layers == layers &&
                        compute_import.producer_command_order == item.command_order;
                    ShaderResource normalized_cube = cube;
                    normalized_cube.format = DataFormat::Unorm16;
                    CHECK(prosper::frontend::live_compute_graphics_import_guest_bytes(
                              normalized_cube, layer_bytes) == normalized_cube.size,
                          "R16 cube import identity covers all six faces, not one decoded face");
                    prosper::frontend::LiveComputeImageImport normalized_import;
                    normalized_imported =
                        prosper::frontend::import_live_compute_storage_image(
                            normalized_cube, normalized_cube.size, normalized_import) &&
                        normalized_import.valid() && normalized_import.native_format == 70u &&
                        normalized_import.vertical_stack_layers == layers &&
                        normalized_import.producer_command_order == item.command_order;
                    ShaderResource float16_cube = cube;
                    float16_cube.format = DataFormat::Float16;
                    prosper::frontend::LiveComputeImageImport float16_import;
                    const bool float16_imported =
                        prosper::frontend::import_live_compute_storage_image(
                            float16_cube, float16_cube.size, float16_import) &&
                        float16_import.valid() && float16_import.native_format == 76u &&
                        float16_import.vertical_stack_layers == layers &&
                        float16_import.producer_command_order == item.command_order;
                    ShaderResource mismatch = cube;
                    --mismatch.depth;
                    prosper::frontend::LiveComputeImageImport rejected;
                    const bool depth_rejected =
                        !prosper::frontend::import_live_compute_storage_image(
                            mismatch, cube.size, rejected);
                    mismatch = cube;
                    mismatch.layer_stride_bytes += sizeof(uint16_t);
                    const bool stride_rejected =
                        !prosper::frontend::import_live_compute_storage_image(
                            mismatch, cube.size, rejected);
                    mismatches_rejected = depth_rejected && stride_rejected && float16_imported;
                    return RenderedFrame{};
                },
                [&](const std::vector<ComputeItem>& items) {
                    return prosper::frontend::execute_live_compute_items(items);
                },
                1, 1);
            CHECK(coverage_proven && ordered.compute_executed && imported,
                  "same-submit cube consumer leases exact six-layer R16 storage for GPU stacking");
            CHECK(normalized_imported,
                  "R16_UNORM cube consumer leases the mutable exact R16_UINT array producer");
            CHECK(mismatches_rejected,
                  "cube-array import rejects depth and layer-stride identity mismatches");
        }
    }
    {
        std::vector<uint8_t> source(W);
        for (uint32_t x = 0; x < W; ++x)
            source[x] = static_cast<uint8_t>(x * 53u + 7u);
        const auto [executed, destination, imported, imported_format] = run_native_typed_copy(
            DataFormat::Uint8, kSpirvImageFormatR8ui, SpirvImageNumericClass::Uint,
            1, 1, source);
        CHECK(executed && destination == source,
              "typed R8_UINT storage copy reflects and executes byte-exactly");
        CHECK(imported && imported_format == 13u, // VK_FORMAT_R8_UINT
              "same-submit graphics consumer leases the exact typed R8_UINT result");
        const auto [alias_executed, alias_destination, alias_imported, alias_format] =
            run_native_typed_copy(DataFormat::Uint8, kSpirvImageFormatR8ui,
                                  SpirvImageNumericClass::Uint, 1, 1, source,
                                  DataFormat::Unorm8);
        CHECK(alias_executed && alias_destination == source,
              "typed R8_UINT storage remains byte-exact for an R8_UNORM graphics consumer");
        CHECK(alias_imported && alias_format == 9u, // VK_FORMAT_R8_UNORM
              "same-submit R8_UNORM consumer leases a mutable view of the exact R8_UINT result");
    }
    {
        std::vector<uint8_t> source(W * 4u);
        for (uint32_t x = 0; x < W; ++x) {
            source[x * 4u + 0u] = static_cast<uint8_t>(x * 53u + 7u);
            source[x * 4u + 1u] = static_cast<uint8_t>(x * 31u + 11u);
            source[x * 4u + 2u] = static_cast<uint8_t>(x * 17u + 13u);
            source[x * 4u + 3u] = static_cast<uint8_t>(255u - x * 3u);
        }
        const auto [executed, destination, imported, imported_format] = run_native_typed_copy(
            DataFormat::Uint8, kSpirvImageFormatRgba8ui, SpirvImageNumericClass::Uint,
            4, 4, source);
        CHECK(executed && destination == source,
              "typed RGBA8_UINT storage copy reflects and executes byte-exactly");
        CHECK(imported && imported_format == 41u, // VK_FORMAT_R8G8B8A8_UINT
              "same-submit graphics consumer leases the exact typed RGBA8_UINT result");
    }
    {
        static const uint32_t store_low_byte_2d[] = {
            0x7e080300u,                         // v4 = x
            0x7e0002ffu, 0x00001234u,            // v0 = value; guest Uint8 store keeps low byte
            0x7e0a0280u,                         // v5 = y = 0
            0xf0200108u, 0x00020004u,            // IMAGE_STORE x to binding 5
            0xbf810000u,
        };
        std::vector<uint8_t> destination(W, 0xa5);
        ShaderResourceTable table;
        ShaderResource output{};
        output.cls = ResourceClass::StorageImage;
        output.img_dim = 1;
        output.binding = 5;
        output.sgpr_base = 8;
        output.format = DataFormat::Uint8;
        output.num_components = 1;
        output.width = W;
        output.height = output.depth = 1;
        output.gpu_addr = reinterpret_cast<uint64_t>(destination.data());
        output.size = W;
        table.resources.push_back(output);
        ComputeShaderConfig config;
        config.user_sgprs.resize(16);
        config.local_x = W;
        config.local_y = config.local_z = 1;
        config.tidig_comp_cnt = 0;
        config.native_storage_format_support =
            native_storage_format_support_bit(DataFormat::Uint8, 1);
        const std::vector<uint32_t> spirv = recompile_compute(
            store_low_byte_2d, std::size(store_low_byte_2d), &table, config);
        ComputeItem item;
        item.spirv = spirv;
        item.resources = std::make_shared<ShaderResourceTable>(table);
        item.launch.threads_x = W;
        item.launch.local_x = W;
        item.launch.groups_x = 1;
        item.launch.threads_y = item.launch.threads_z = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x5905a3u;
        CHECK(!spirv.empty() && prosper::frontend::execute_live_compute_items({item}) &&
                  std::all_of(destination.begin(), destination.end(),
                              [](uint8_t value) { return value == 0x34u; }),
              "typed R8_UINT storage truncates a 32-bit guest value to its exact low byte");
    }

    // GTA V gameplay's exact IMAGE_LOAD_MIP / IMAGE_STORE_MIP packets. The live backend has one
    // materialized level for these audited descriptors; the shader marker proves only the mip
    // operand is zero. Exercise all four VDATA components of 0x2042f49800's dmask-f load, preserve a
    // distinct 2D-array slice, and execute the NSA dmask-x store through its StorageImage contract.
    const uint32_t gta_load_mip_copy_2d[] = {
        0x7e080300u,                         // v4 = x (save shell input v0)
        0x7e020280u,                         // v1 = y = 0
        0x7e040207u,                         // v2 = mip = s7 = 0
        0x7e0a0280u,                         // v5 = destination y = 0
        0xf0043f08u, 0x00050000u,            // IMAGE_LOAD_MIP v[0:3], [v0,v1,v2], s[20:27]
        0xbf8c3f70u,
        0xf0200f08u, 0x00020004u,            // IMAGE_STORE v[0:3], [v4,v5], s[8:15]
        0xbf810000u,
    };
    std::vector<uint8_t> mip2d_dst(W * 4, 0x6b);
    ShaderResourceTable mip2d_rt = irt;
    for (ShaderResource& resource : mip2d_rt.resources) {
        if (resource.binding == 4) {
            resource.cls = ResourceClass::Texture;
            resource.sgpr_base = 20;
            resource.fetch_pc = 4;
            resource.img_dim = 1;
            resource.proven_zero_mip = true;
        } else if (resource.binding == 5) {
            resource.img_dim = 1; // exact destination IMAGE_STORE has DIM=2D
            resource.gpu_addr = reinterpret_cast<uint64_t>(mip2d_dst.data());
            resource.size = static_cast<uint32_t>(mip2d_dst.size());
        }
    }
    const std::vector<uint32_t> gta_load_mip_2d_spirv = recompile_valu(
        gta_load_mip_copy_2d, std::size(gta_load_mip_copy_2d), 1, 0, &mip2d_rt);
    CHECK(!gta_load_mip_2d_spirv.empty(),
          "GTA V dmask-xyzw IMAGE_LOAD_MIP 2D recompiles for live execution");
    if (!gta_load_mip_2d_spirv.empty()) {
        ComputeItem item;
        item.spirv = gta_load_mip_2d_spirv;
        item.resources = std::make_shared<ShaderResourceTable>(mip2d_rt);
        item.launch.threads_x = W; item.launch.local_x = 64; item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x248149800ull;
        CHECK(prosper::frontend::execute_live_compute_items({item}),
              "live backend executes GTA V's dmask-xyzw IMAGE_LOAD_MIP 2D");
        CHECK(mip2d_dst == img_src,
              "IMAGE_LOAD_MIP 2D preserves all four returned channels at level zero");
    }

    const uint32_t gta_load_mip_copy_2da[] = {
        0x7e080300u,                         // v4 = x
        0x7e020280u,                         // v1 = y = 0
        0x7e040281u,                         // v2 = array slice 1
        0x7e060206u,                         // v3 = mip = s6 = 0
        0x7e0a0280u,                         // v5 = destination y = 0
        0xf0043128u, 0x00050000u,            // IMAGE_LOAD_MIP 2D_ARRAY dmask:x
        0xbf8c3f70u,
        0xf0200108u, 0x00020004u,            // IMAGE_STORE x channel
        0xbf810000u,
    };
    std::vector<uint32_t> mip_array_src(W * 2u);
    std::vector<uint32_t> mip_array_dst(W, 0x6d6d6d6du);
    for (uint32_t i = 0; i < W; ++i) {
        mip_array_src[i] = i * 13u + 7u;
        mip_array_src[W + i] = i * 65537u + 0x120034u;
    }
    ShaderResourceTable mip_array_rt = irt;
    for (ShaderResource& resource : mip_array_rt.resources) {
        if (resource.binding == 4) {
            resource.cls = ResourceClass::Texture;
            resource.sgpr_base = 20;
            resource.fetch_pc = 5;
            resource.img_dim = 5;
            resource.depth = 2;
            resource.format = DataFormat::Uint32;
            resource.num_components = 1;
            resource.gpu_addr = reinterpret_cast<uint64_t>(mip_array_src.data());
            resource.size = static_cast<uint32_t>(
                mip_array_src.size() * sizeof(uint32_t));
            resource.layer_stride_bytes = W * 4u;
            resource.layer_mip_offset_bytes = 0;
            resource.proven_zero_mip = true;
        } else if (resource.binding == 5) {
            resource.img_dim = 1; // exact destination IMAGE_STORE has DIM=2D
            resource.format = DataFormat::Uint32;
            resource.num_components = 1;
            resource.gpu_addr = reinterpret_cast<uint64_t>(mip_array_dst.data());
            resource.size = static_cast<uint32_t>(
                mip_array_dst.size() * sizeof(uint32_t));
        }
    }
    const std::vector<uint32_t> gta_load_mip_array_spirv = recompile_valu(
        gta_load_mip_copy_2da, std::size(gta_load_mip_copy_2da), 1, 0, &mip_array_rt);
    CHECK(!gta_load_mip_array_spirv.empty(),
          "GTA V IMAGE_LOAD_MIP 2D_ARRAY recompiles with its proven level-zero marker");
    if (!gta_load_mip_array_spirv.empty()) {
        ComputeItem item;
        item.spirv = gta_load_mip_array_spirv;
        item.resources = std::make_shared<ShaderResourceTable>(mip_array_rt);
        item.launch.threads_x = W; item.launch.local_x = 64; item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x24814a600ull;
        CHECK(prosper::frontend::execute_live_compute_items({item}),
              "live backend executes GTA V's IMAGE_LOAD_MIP 2D_ARRAY");
        CHECK(std::equal(mip_array_dst.begin(), mip_array_dst.end(),
                         mip_array_src.begin() + W),
              "IMAGE_LOAD_MIP 2D_ARRAY fetches distinct slice one instead of dropping the layer");
    }

    // #3048: the DYNAMIC mip operand. Sonic Frontiers' three scene-target-width stage kernels issue
    // IMAGE_LOAD_MIP against a 12-level 2048x2048 surface with the mip NOT provably zero, so the
    // zero-specialization arms above can never admit them -- and emitting a Lod against the old
    // hard-coded `mipLevels = 1` image would have fetched a level that does not exist. The compute
    // backend now materializes the chain the T# declares, uploading each level from the guest's own
    // tiled mip offsets, and the operand reaches OpImageFetch's Lod. The discriminator is that the
    // dispatch returns LEVEL ONE's texels: a single-level image cannot produce them.
    {
        constexpr uint32_t CHAIN_W = 256, CHAIN_H = 256, CHAIN_BPE = 4, CHAIN_MAX_MIP = 8;
        const uint32_t chain_tile = static_cast<uint32_t>(TileMode::Sw64KbRX);
        const TiledMipLevelLayout chain_level0 = tiled_mip_level_layout(
            CHAIN_W, CHAIN_H, CHAIN_BPE, chain_tile, CHAIN_MAX_MIP, 0);
        const TiledMipLevelLayout chain_level1 = tiled_mip_level_layout(
            CHAIN_W, CHAIN_H, CHAIN_BPE, chain_tile, CHAIN_MAX_MIP, 1);
        const size_t chain_bytes = tiled_mip_chain_bytes(
            CHAIN_W, CHAIN_H, CHAIN_BPE, chain_tile, CHAIN_MAX_MIP);
        CHECK(chain_level0.supported && !chain_level0.in_tail && chain_level1.supported &&
                  !chain_level1.in_tail && chain_bytes != 0,
              "the dynamic-mip fixture places levels zero and one outside the shared mip tail");

        std::vector<uint8_t> chain_allocation(chain_bytes, 0x5a);
        std::vector<uint32_t> chain_level0_linear(static_cast<size_t>(CHAIN_W) * CHAIN_H);
        std::vector<uint32_t> chain_level1_linear(
            static_cast<size_t>(CHAIN_W / 2u) * (CHAIN_H / 2u));
        for (size_t i = 0; i < chain_level0_linear.size(); ++i)
            chain_level0_linear[i] = static_cast<uint32_t>(i * 2654435761u + 0x11111111u);
        for (size_t i = 0; i < chain_level1_linear.size(); ++i)
            chain_level1_linear[i] = static_cast<uint32_t>(i * 40503u + 0xa5a50000u);
        tile_surface(chain_allocation.data() + chain_level0.byte_offset,
                     reinterpret_cast<const uint8_t*>(chain_level0_linear.data()),
                     CHAIN_W, CHAIN_H, chain_tile, 0, CHAIN_BPE);
        tile_surface(chain_allocation.data() + chain_level1.byte_offset,
                     reinterpret_cast<const uint8_t*>(chain_level1_linear.data()),
                     CHAIN_W / 2u, CHAIN_H / 2u, chain_tile, 0, CHAIN_BPE);

        // word0 f0040108 = IMAGE_LOAD_MIP (opcode 1) dmask:x dim:2D with UNRM and GLC CLEAR -- the
        // exact modifier shape Sonic Frontiers emits, which `rdna2_mimg_zero_mip_shape` rejects.
        const uint32_t dynamic_mip_program[] = {
            0x7e080300u,                     // v4 = x (save the shell input before v0 is reused)
            0x7e020280u,                     // v1 = y = 0
            0x7e040281u,                     // v2 = mip = inline constant 1 (never provably zero)
            0x7e0a0280u,                     // v5 = destination y = 0
            0xf0040108u, 0x00050000u,        // IMAGE_LOAD_MIP v0, [v0,v1,v2], s[20:27]
            0xbf8c3f70u,                     // s_waitcnt
            0xf0200108u, 0x00020004u,        // IMAGE_STORE v0, [v4,v5], s[8:15]
            0xbf810000u,                     // s_endpgm
        };
        std::vector<uint32_t> chain_dst(W, 0x77777777u);
        ShaderResourceTable chain_rt = irt;
        for (ShaderResource& resource : chain_rt.resources) {
            if (resource.binding == 4) {
                resource.cls = ResourceClass::Texture;
                resource.sgpr_base = 20;
                resource.fetch_pc = 4;
                resource.img_dim = 1;
                resource.format = DataFormat::Uint32;
                resource.num_components = 1;
                resource.width = CHAIN_W;
                resource.height = CHAIN_H;
                resource.depth = 1;
                resource.tile_mode = chain_tile;
                resource.declared_mip_levels = CHAIN_MAX_MIP + 1u;
                resource.mip_chain_element_width = CHAIN_W;
                resource.mip_chain_element_height = CHAIN_H;
                resource.mip_chain_bytes_per_block = CHAIN_BPE;
                resource.mip_chain_max_level = CHAIN_MAX_MIP;
                resource.mip_chain_base_level = 0;
                resource.proven_zero_mip = false;   // the whole point: the operand is a real value
                resource.gpu_addr = reinterpret_cast<uint64_t>(
                    chain_allocation.data() + chain_level0.byte_offset);
                resource.size = static_cast<uint32_t>(
                    tiled_surface_bytes(CHAIN_W, CHAIN_H, chain_tile, 0, CHAIN_BPE));
            } else if (resource.binding == 5) {
                resource.img_dim = 1;   // exact destination IMAGE_STORE has DIM=2D
                resource.format = DataFormat::Uint32;
                resource.num_components = 1;
                resource.gpu_addr = reinterpret_cast<uint64_t>(chain_dst.data());
                resource.size = static_cast<uint32_t>(chain_dst.size() * sizeof(uint32_t));
            }
        }
        const std::vector<uint32_t> chain_spirv = recompile_valu(
            dynamic_mip_program, std::size(dynamic_mip_program), 1, 0, &chain_rt);
        CHECK(!chain_spirv.empty(),
              "IMAGE_LOAD_MIP with a runtime mip operand recompiles once the declared chain is "
              "materialized (#3048)");
        if (!chain_spirv.empty()) {
            ComputeItem item;
            item.spirv = chain_spirv;
            item.resources = std::make_shared<ShaderResourceTable>(chain_rt);
            item.launch.threads_x = W; item.launch.local_x = 64; item.launch.groups_x = 1;
            item.launch.local_y = item.launch.local_z = 1;
            item.launch.groups_y = item.launch.groups_z = 1;
            item.code_addr = 0x2005714000ull;
            CHECK(prosper::frontend::execute_live_compute_items({item}),
                  "live backend executes a dynamic-mip IMAGE_LOAD_MIP dispatch");
            CHECK(std::equal(chain_dst.begin(), chain_dst.end(), chain_level1_linear.begin()),
                  "IMAGE_LOAD_MIP at a runtime mip returns LEVEL ONE's own guest texels");
            // The two arms below are FIXTURE-VALIDITY guards, not discriminators, and the
            // difference is measured rather than argued: with the backend reverted to
            // `ici.mipLevels = 1` the assertion above reddens while BOTH of these stay green.
            // So the destination is not left unwritten under that mutant -- it is written with
            // something that is neither level one nor the sentinel. What they rule out is a
            // fixture that could pass vacuously: level one being byte-identical to level zero,
            // and the dispatch never running at all. The discriminator is the level-one equality
            // itself, and behind it the fact that this program only recompiles through the
            // dynamic-LOD path. Do not read either arm as evidence about the mutant.
            CHECK(!std::equal(chain_dst.begin(), chain_dst.end(), chain_level0_linear.begin()),
                  "level one is distinguishable from level zero in this fixture");
            CHECK(std::none_of(chain_dst.begin(), chain_dst.end(),
                               [](uint32_t value) { return value == 0x77777777u; }),
                  "every destination texel was actually written by the dispatch");
        }
    }

    // #3134: the SAME dynamic-mip lowering reached through the NSA address encoding. Stray's
    // (PPSA02101) title-screen fragment issues `image_load_mip v[5:7], [v0, v42, v5], s[32:39]
    // dmask:0x7 dim:SQ_RSRC_IMG_2D` -- three dwords, because NSA spells each address after the
    // first as one byte of an extra dword instead of taking consecutive VGPRs from VADDR. That is
    // an ADDRESS ENCODING, not a different operation, and the shape predicate could not read it,
    // so the whole stage was rejected `mode=unresolved-operand`.
    //
    // The discriminator is deliberately built so a consecutive-VGPR reading CANNOT pass it: the
    // mip lives in v7 (value 1) while v2 -- the register the old arithmetic would have named --
    // holds 0. A regression that read v2 would fetch level ZERO and redden the level-one equality.
    {
        constexpr uint32_t NSA_W = 256, NSA_H = 256, NSA_BPE = 4, NSA_MAX_MIP = 8;
        const uint32_t nsa_tile = static_cast<uint32_t>(TileMode::Sw64KbRX);
        const TiledMipLevelLayout nsa_level0 = tiled_mip_level_layout(
            NSA_W, NSA_H, NSA_BPE, nsa_tile, NSA_MAX_MIP, 0);
        const TiledMipLevelLayout nsa_level1 = tiled_mip_level_layout(
            NSA_W, NSA_H, NSA_BPE, nsa_tile, NSA_MAX_MIP, 1);
        const size_t nsa_bytes = tiled_mip_chain_bytes(
            NSA_W, NSA_H, NSA_BPE, nsa_tile, NSA_MAX_MIP);
        CHECK(nsa_level0.supported && !nsa_level0.in_tail && nsa_level1.supported &&
                  !nsa_level1.in_tail && nsa_bytes != 0,
              "the NSA fixture places levels zero and one outside the shared mip tail");

        std::vector<uint8_t> nsa_allocation(nsa_bytes, 0x6b);
        std::vector<uint32_t> nsa_level0_linear(static_cast<size_t>(NSA_W) * NSA_H);
        std::vector<uint32_t> nsa_level1_linear(
            static_cast<size_t>(NSA_W / 2u) * (NSA_H / 2u));
        for (size_t i = 0; i < nsa_level0_linear.size(); ++i)
            nsa_level0_linear[i] = static_cast<uint32_t>(i * 1103515245u + 0x33333333u);
        for (size_t i = 0; i < nsa_level1_linear.size(); ++i)
            nsa_level1_linear[i] = static_cast<uint32_t>(i * 22695477u + 0x6b6b0000u);
        tile_surface(nsa_allocation.data() + nsa_level0.byte_offset,
                     reinterpret_cast<const uint8_t*>(nsa_level0_linear.data()),
                     NSA_W, NSA_H, nsa_tile, 0, NSA_BPE);
        tile_surface(nsa_allocation.data() + nsa_level1.byte_offset,
                     reinterpret_cast<const uint8_t*>(nsa_level1_linear.data()),
                     NSA_W / 2u, NSA_H / 2u, nsa_tile, 0, NSA_BPE);

        // Every word below is llvm-mc gfx1030 output for the instruction in its comment.
        const uint32_t nsa_mip_program[] = {
            0x7e080300u,                                  // v4 = x (shell input, saved)
            0x7e020280u,                                  // v1 = y = 0
            0x7e040280u,                                  // v2 = 0  <- a consecutive read lands here
            0x7e0e0281u,                                  // v7 = mip = 1 <- the NSA byte's register
            0x7e0a0280u,                                  // v5 = destination y = 0
            0xf004010au, 0x00050000u, 0x00000701u,        // IMAGE_LOAD_MIP v0, [v0,v1,v7], s[20:27]
            0xbf8c3f70u,                                  // s_waitcnt
            0xf0200108u, 0x00020004u,                     // IMAGE_STORE v0, [v4,v5], s[8:15]
            0xbf810000u,                                  // s_endpgm
        };
        {
            // The encoding claim this arm rests on, checked against the decoder rather than
            // assumed from the comment: three dwords, NSA=1, and the mip named by the SECOND
            // address byte.
            uint32_t decoded_mip = UINT32_MAX;
            const Rdna2Inst decoded = rdna2_decode_one(&nsa_mip_program[5], 3);
            CHECK(decoded.fmt == Rdna2Format::MIMG && decoded.opcode == 0x01u &&
                      decoded.mimg_nsa == 1u && decoded.len_dwords == 3u &&
                      rdna2_mimg_dynamic_mip_shape(decoded, &decoded_mip) && decoded_mip == 7u,
                  "#3134: the fixture's NSA packet names v7 -- not v2 -- as its mip operand");
        }
        std::vector<uint32_t> nsa_dst(W, 0x44444444u);
        ShaderResourceTable nsa_rt = irt;
        for (ShaderResource& resource : nsa_rt.resources) {
            if (resource.binding == 4) {
                resource.cls = ResourceClass::Texture;
                resource.sgpr_base = 20;
                resource.fetch_pc = 5;
                resource.img_dim = 1;
                resource.format = DataFormat::Uint32;
                resource.num_components = 1;
                resource.width = NSA_W;
                resource.height = NSA_H;
                resource.depth = 1;
                resource.tile_mode = nsa_tile;
                resource.declared_mip_levels = NSA_MAX_MIP + 1u;
                resource.mip_chain_element_width = NSA_W;
                resource.mip_chain_element_height = NSA_H;
                resource.mip_chain_bytes_per_block = NSA_BPE;
                resource.mip_chain_max_level = NSA_MAX_MIP;
                resource.mip_chain_base_level = 0;
                resource.proven_zero_mip = false;
                resource.gpu_addr = reinterpret_cast<uint64_t>(
                    nsa_allocation.data() + nsa_level0.byte_offset);
                resource.size = static_cast<uint32_t>(
                    tiled_surface_bytes(NSA_W, NSA_H, nsa_tile, 0, NSA_BPE));
            } else if (resource.binding == 5) {
                resource.img_dim = 1;
                resource.format = DataFormat::Uint32;
                resource.num_components = 1;
                resource.gpu_addr = reinterpret_cast<uint64_t>(nsa_dst.data());
                resource.size = static_cast<uint32_t>(nsa_dst.size() * sizeof(uint32_t));
            }
        }
        const std::vector<uint32_t> nsa_spirv = recompile_valu(
            nsa_mip_program, std::size(nsa_mip_program), 1, 0, &nsa_rt);
        CHECK(!nsa_spirv.empty(),
              "#3134: an NSA-encoded IMAGE_LOAD_MIP recompiles instead of rejecting the stage");
        if (!nsa_spirv.empty()) {
            ComputeItem item;
            item.spirv = nsa_spirv;
            item.resources = std::make_shared<ShaderResourceTable>(nsa_rt);
            item.launch.threads_x = W; item.launch.local_x = 64; item.launch.groups_x = 1;
            item.launch.local_y = item.launch.local_z = 1;
            item.launch.groups_y = item.launch.groups_z = 1;
            item.code_addr = 0x30be800000ull;   // Stray's own fragment address, as a label
            CHECK(prosper::frontend::execute_live_compute_items({item}),
                  "#3134: the live backend executes an NSA-addressed dynamic-mip dispatch");
            CHECK(std::equal(nsa_dst.begin(), nsa_dst.end(), nsa_level1_linear.begin()),
                  "#3134: the NSA packet's mip byte selects LEVEL ONE's own guest texels");
            // Fixture-validity guards, not discriminators -- the same distinction #3048's block
            // draws. Level one must be distinguishable from level zero, and the dispatch must
            // actually have written every texel, or the equality above could pass vacuously.
            CHECK(!std::equal(nsa_dst.begin(), nsa_dst.end(), nsa_level0_linear.begin()),
                  "#3134: level one is distinguishable from level zero in the NSA fixture");
            CHECK(std::none_of(nsa_dst.begin(), nsa_dst.end(),
                               [](uint32_t value) { return value == 0x44444444u; }),
                  "#3134: every NSA destination texel was actually written by the dispatch");
        }
    }

    // #3134: a chain whose SELECTED level is itself packed in the shared mip tail. Stray's
    // resource is a 32x32 six-level pyramid, which at 4 bytes per texel is smaller than one 64 KiB
    // macroblock -- so AddrLib packs every level, level zero included, into the allocation's tail
    // and leaves `gpu_addr` at the allocation base. `shader_resource_mip_chain_plan` used to refuse
    // that placement outright, which refused the whole small-texture class rather than an exotic
    // corner. 64x64 is used here (not 32x32) only so level one is 32 texels wide, which the
    // dispatch below covers exactly; the tail geometry is identical.
    {
        constexpr uint32_t TAIL_W = 64, TAIL_H = 64, TAIL_BPE = 4, TAIL_MAX_MIP = 6;
        constexpr uint32_t TAIL_THREADS = 32;   // = level one's width
        const uint32_t tail_tile = static_cast<uint32_t>(TileMode::Sw64KbRX);
        const TiledMipLevelLayout tail_level0 = tiled_mip_level_layout(
            TAIL_W, TAIL_H, TAIL_BPE, tail_tile, TAIL_MAX_MIP, 0);
        const TiledMipLevelLayout tail_level1 = tiled_mip_level_layout(
            TAIL_W, TAIL_H, TAIL_BPE, tail_tile, TAIL_MAX_MIP, 1);
        const size_t tail_bytes = tiled_mip_chain_bytes(
            TAIL_W, TAIL_H, TAIL_BPE, tail_tile, TAIL_MAX_MIP);
        CHECK(tail_level0.supported && tail_level0.in_tail && tail_level1.supported &&
                  tail_level1.in_tail && tail_bytes != 0 &&
                  tail_level0.tail_block_bytes == tail_bytes,
              "#3134: the tail fixture really packs level zero AND level one into one block");

        std::vector<uint8_t> tail_allocation(tail_bytes, 0x91);
        std::vector<uint32_t> tail_level0_linear(static_cast<size_t>(TAIL_W) * TAIL_H);
        std::vector<uint32_t> tail_level1_linear(
            static_cast<size_t>(TAIL_W / 2u) * (TAIL_H / 2u));
        for (size_t i = 0; i < tail_level0_linear.size(); ++i)
            tail_level0_linear[i] = static_cast<uint32_t>(i * 2891336453u + 0x19191919u);
        for (size_t i = 0; i < tail_level1_linear.size(); ++i)
            tail_level1_linear[i] = static_cast<uint32_t>(i * 374761393u + 0x91910000u);
        // Tail levels share one block and are addressed by element coordinate inside it, so they
        // are written with the sibling-preserving level writer rather than by offset.
        tile_surface_level(tail_allocation.data(), tail_allocation.size(),
                           reinterpret_cast<const uint8_t*>(tail_level0_linear.data()),
                           TAIL_W, TAIL_H, tail_tile, TAIL_BPE,
                           tail_level0.tail_x, tail_level0.tail_y);
        tile_surface_level(tail_allocation.data(), tail_allocation.size(),
                           reinterpret_cast<const uint8_t*>(tail_level1_linear.data()),
                           TAIL_W / 2u, TAIL_H / 2u, tail_tile, TAIL_BPE,
                           tail_level1.tail_x, tail_level1.tail_y);

        const uint32_t tail_mip_program[] = {
            0x7e080300u,                                  // v4 = x
            0x7e020280u,                                  // v1 = y = 0
            0x7e040281u,                                  // v2 = mip = 1
            0x7e0a0280u,                                  // v5 = destination y = 0
            0xf0040108u, 0x00050000u,                     // IMAGE_LOAD_MIP v0, [v0,v1,v2], s[20:27]
            0xbf8c3f70u,                                  // s_waitcnt
            0xf0200108u, 0x00020004u,                     // IMAGE_STORE v0, [v4,v5], s[8:15]
            0xbf810000u,                                  // s_endpgm
        };
        std::vector<uint32_t> tail_dst(TAIL_THREADS, 0x27272727u);
        ShaderResourceTable tail_rt = irt;
        for (ShaderResource& resource : tail_rt.resources) {
            if (resource.binding == 4) {
                resource.cls = ResourceClass::Texture;
                resource.sgpr_base = 20;
                resource.fetch_pc = 4;
                resource.img_dim = 1;
                resource.format = DataFormat::Uint32;
                resource.num_components = 1;
                resource.width = TAIL_W;
                resource.height = TAIL_H;
                resource.depth = 1;
                resource.tile_mode = tail_tile;
                resource.declared_mip_levels = TAIL_MAX_MIP + 1u;
                resource.mip_chain_element_width = TAIL_W;
                resource.mip_chain_element_height = TAIL_H;
                resource.mip_chain_bytes_per_block = TAIL_BPE;
                resource.mip_chain_max_level = TAIL_MAX_MIP;
                resource.mip_chain_base_level = 0;
                resource.proven_zero_mip = false;
                // The descriptor's own published tail placement -- gpu_addr is NOT shifted onto
                // the selected level, because a tail level owns no disjoint byte range.
                resource.in_mip_tail = true;
                resource.mip_tail_offset = static_cast<uint32_t>(tail_level0.byte_offset);
                resource.mip_tail_bytes = tail_level0.tail_block_bytes;
                resource.mip_tail_x = tail_level0.tail_x;
                resource.mip_tail_y = tail_level0.tail_y;
                resource.gpu_addr = reinterpret_cast<uint64_t>(tail_allocation.data());
                resource.size = static_cast<uint32_t>(tail_bytes);
            } else if (resource.binding == 5) {
                resource.img_dim = 1;
                resource.format = DataFormat::Uint32;
                resource.num_components = 1;
                // The destination is only as wide as the dispatch, because level one is 32 texels
                // across: a wider one would leave the tail of the row unwritten and out of bounds.
                resource.width = TAIL_THREADS;
                resource.height = 1;
                resource.gpu_addr = reinterpret_cast<uint64_t>(tail_dst.data());
                resource.size = static_cast<uint32_t>(tail_dst.size() * sizeof(uint32_t));
            }
        }
        const std::vector<uint32_t> tail_spirv = recompile_valu(
            tail_mip_program, std::size(tail_mip_program), 1, 0, &tail_rt);
        CHECK(!tail_spirv.empty(),
              "#3134: IMAGE_LOAD_MIP recompiles against a tail-packed declared chain");
        if (!tail_spirv.empty()) {
            ComputeItem item;
            item.spirv = tail_spirv;
            item.resources = std::make_shared<ShaderResourceTable>(tail_rt);
            item.launch.threads_x = TAIL_THREADS; item.launch.local_x = TAIL_THREADS;
            item.launch.groups_x = 1;
            item.launch.local_y = item.launch.local_z = 1;
            item.launch.groups_y = item.launch.groups_z = 1;
            item.code_addr = 0x308e090000ull;   // Stray's own resource address, as a label
            CHECK(prosper::frontend::execute_live_compute_items({item}),
                  "#3134: the live backend executes a dispatch over a tail-packed chain");
            CHECK(std::equal(tail_dst.begin(), tail_dst.end(), tail_level1_linear.begin()),
                  "#3134: a tail-packed level one returns its OWN guest texels");
            CHECK(!std::equal(tail_dst.begin(), tail_dst.end(), tail_level0_linear.begin()),
                  "#3134: level one is distinguishable from level zero in the tail fixture");
            CHECK(std::none_of(tail_dst.begin(), tail_dst.end(),
                               [](uint32_t value) { return value == 0x27272727u; }),
                  "#3134: every tail destination texel was actually written by the dispatch");
        }
    }

    // #3048: the SAME dynamic-mip lowering at DMASK 0x3. `rdna2_mimg_dynamic_mip_shape` widened
    // DMASK from the zero-mip shape's {0x1, 0xf} to anything, and Sonic Frontiers' kernels issue
    // exactly 0x3 against a two-channel surface -- so the two-component result mask, and the
    // register the second component lands in, need their own executed arm. Word0 here is byte-equal
    // to the word the [mimg-mip] diagnostic reports for `0x2005714000`: `f0040308`.
    {
        constexpr uint32_t RG_W = 256, RG_H = 256, RG_BPE = 8, RG_MAX_MIP = 8;
        const uint32_t rg_tile = static_cast<uint32_t>(TileMode::Sw64KbRX);
        const TiledMipLevelLayout rg_level0 = tiled_mip_level_layout(
            RG_W, RG_H, RG_BPE, rg_tile, RG_MAX_MIP, 0);
        const TiledMipLevelLayout rg_level1 = tiled_mip_level_layout(
            RG_W, RG_H, RG_BPE, rg_tile, RG_MAX_MIP, 1);
        const size_t rg_bytes = tiled_mip_chain_bytes(
            RG_W, RG_H, RG_BPE, rg_tile, RG_MAX_MIP);
        CHECK(rg_level0.supported && !rg_level0.in_tail && rg_level1.supported &&
                  !rg_level1.in_tail && rg_bytes != 0,
              "the dmask:0x3 fixture places levels zero and one outside the shared mip tail");

        std::vector<uint8_t> rg_allocation(rg_bytes, 0x3c);
        std::vector<uint32_t> rg_level0_linear(static_cast<size_t>(RG_W) * RG_H * 2u);
        std::vector<uint32_t> rg_level1_linear(
            static_cast<size_t>(RG_W / 2u) * (RG_H / 2u) * 2u);
        for (size_t i = 0; i < rg_level0_linear.size(); ++i)
            rg_level0_linear[i] = static_cast<uint32_t>(i * 2246822519u + 0x22222222u);
        for (size_t i = 0; i < rg_level1_linear.size(); ++i)
            rg_level1_linear[i] = static_cast<uint32_t>(i * 668265263u + 0x5c5c0000u);
        tile_surface(rg_allocation.data() + rg_level0.byte_offset,
                     reinterpret_cast<const uint8_t*>(rg_level0_linear.data()),
                     RG_W, RG_H, rg_tile, 0, RG_BPE);
        tile_surface(rg_allocation.data() + rg_level1.byte_offset,
                     reinterpret_cast<const uint8_t*>(rg_level1_linear.data()),
                     RG_W / 2u, RG_H / 2u, rg_tile, 0, RG_BPE);

        // VDATA is v[0:1] for dmask 0x3, so the coordinates live at v2 upward and the store reads
        // the SECOND returned component -- the one a dmask-0x1 arm can never observe.
        const uint32_t dmask3_program[] = {
            0x7e040300u,                     // v2 = x (VADDR base)
            0x7e060280u,                     // v3 = y = 0
            0x7e080281u,                     // v4 = mip = inline constant 1
            0x7e0a0300u,                     // v5 = destination x
            0x7e0c0280u,                     // v6 = destination y = 0
            0xf0040308u, 0x00050002u,        // IMAGE_LOAD_MIP dmask:0x3 v[0:1], [v2,v3,v4], s[20:27]
            0xbf8c3f70u,                     // s_waitcnt
            0xf0200108u, 0x00020105u,        // IMAGE_STORE dmask:0x1 v1, [v5,v6], s[8:15]
            0xbf810000u,                     // s_endpgm
        };
        std::vector<uint32_t> rg_dst(W, 0x66666666u);
        ShaderResourceTable rg_rt = irt;
        for (ShaderResource& resource : rg_rt.resources) {
            if (resource.binding == 4) {
                resource.cls = ResourceClass::Texture;
                resource.sgpr_base = 20;
                resource.fetch_pc = 5;
                resource.img_dim = 1;
                resource.format = DataFormat::Uint32;
                resource.num_components = 2;      // Sonic's surface is two-channel at 8 B/texel
                resource.width = RG_W;
                resource.height = RG_H;
                resource.depth = 1;
                resource.tile_mode = rg_tile;
                resource.declared_mip_levels = RG_MAX_MIP + 1u;
                resource.mip_chain_element_width = RG_W;
                resource.mip_chain_element_height = RG_H;
                resource.mip_chain_bytes_per_block = RG_BPE;
                resource.mip_chain_max_level = RG_MAX_MIP;
                resource.mip_chain_base_level = 0;
                resource.proven_zero_mip = false;
                resource.gpu_addr = reinterpret_cast<uint64_t>(
                    rg_allocation.data() + rg_level0.byte_offset);
                resource.size = static_cast<uint32_t>(
                    tiled_surface_bytes(RG_W, RG_H, rg_tile, 0, RG_BPE));
            } else if (resource.binding == 5) {
                resource.img_dim = 1;
                resource.format = DataFormat::Uint32;
                resource.num_components = 1;
                resource.gpu_addr = reinterpret_cast<uint64_t>(rg_dst.data());
                resource.size = static_cast<uint32_t>(rg_dst.size() * sizeof(uint32_t));
            }
        }
        const std::vector<uint32_t> rg_spirv = recompile_valu(
            dmask3_program, std::size(dmask3_program), 1, 0, &rg_rt);
        CHECK(!rg_spirv.empty(),
              "#3048: a dmask:0x3 IMAGE_LOAD_MIP with a runtime mip recompiles");
        if (!rg_spirv.empty()) {
            ComputeItem item;
            item.spirv = rg_spirv;
            item.resources = std::make_shared<ShaderResourceTable>(rg_rt);
            item.launch.threads_x = W; item.launch.local_x = 64; item.launch.groups_x = 1;
            item.launch.local_y = item.launch.local_z = 1;
            item.launch.groups_y = item.launch.groups_z = 1;
            item.code_addr = 0x200571bd00ull;
            CHECK(prosper::frontend::execute_live_compute_items({item}),
                  "#3048: live backend executes the dmask:0x3 dynamic-mip dispatch");
            bool second_component_exact = true;
            for (uint32_t x = 0; x < W; ++x)
                second_component_exact &=
                    rg_dst[x] == rg_level1_linear[static_cast<size_t>(x) * 2u + 1u];
            CHECK(second_component_exact,
                  "#3048: dmask:0x3 returns level one's SECOND component in the second VGPR");
            CHECK(std::none_of(rg_dst.begin(), rg_dst.end(),
                               [](uint32_t value) { return value == 0x66666666u; }),
                  "#3048: every dmask:0x3 destination texel was actually written");
        }
    }

    const uint32_t gta_store_mip_2d[] = {
        0x7e080300u,                         // v4 = x, while v0 remains stored data
        0x7e060280u,                         // v3 = y = 0
        0x7e0a0206u,                         // v5 = mip = s6 = 0
        0xf024310au, 0x00030004u, 0x00000503u,
        0xbf810000u,
    };
    std::vector<uint32_t> store_mip_dst(W, 0xdeadbeefu);
    ShaderResourceTable store_mip_rt = irt;
    store_mip_rt.resources.erase(
        std::remove_if(store_mip_rt.resources.begin(), store_mip_rt.resources.end(),
                       [](const ShaderResource& resource) {
                           return resource.cls == ResourceClass::StorageImage;
                       }),
        store_mip_rt.resources.end());
    ShaderResource store_mip_image{};
    store_mip_image.cls = ResourceClass::StorageImage;
    store_mip_image.binding = 4; store_mip_image.fetch_pc = 3;
    store_mip_image.sgpr_base = 12; store_mip_image.img_dim = 1;
    store_mip_image.format = DataFormat::Uint32; store_mip_image.num_components = 1;
    store_mip_image.width = W; store_mip_image.height = store_mip_image.depth = 1;
    store_mip_image.gpu_addr = reinterpret_cast<uint64_t>(store_mip_dst.data());
    store_mip_image.size = static_cast<uint32_t>(store_mip_dst.size() * sizeof(uint32_t));
    store_mip_image.proven_zero_mip = true;
    store_mip_rt.resources.push_back(store_mip_image);
    const std::vector<uint32_t> gta_store_mip_spirv = recompile_valu(
        gta_store_mip_2d, std::size(gta_store_mip_2d), 1, 0, &store_mip_rt);
    CHECK(!gta_store_mip_spirv.empty(),
          "GTA V IMAGE_STORE_MIP 2D recompiles with StorageImage classification");
    if (!gta_store_mip_spirv.empty()) {
        ComputeItem item;
        item.spirv = gta_store_mip_spirv;
        item.resources = std::make_shared<ShaderResourceTable>(store_mip_rt);
        item.launch.threads_x = W; item.launch.local_x = 64; item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x248149a00ull;
        CHECK(prosper::frontend::execute_live_compute_items({item}),
              "live backend executes GTA V's IMAGE_STORE_MIP 2D");
        CHECK(store_mip_dst == lane_index,
              "IMAGE_STORE_MIP writes every lane's dword to level zero");
    }

    const uint32_t gta_store_mip_xyzw_2d[] = {
        0x7e0a0300u,                         // v5 = x, while v0 remains channel X data
        0x7e080280u,                         // v4 = y = 0
        0x7e020281u,                         // v1 = channel Y = 1
        0x7e040282u,                         // v2 = channel Z = 2
        0x7e060283u,                         // v3 = channel W = 3
        0x7e0c0206u,                         // v6 = mip = s6 = 0
        0xf0243f0au, 0x00030005u, 0x00000604u, // live 0x2042f49800 packet
        0xbf810000u,
    };
    std::vector<uint8_t> store_mip_rgba_dst(W * 4u, 0xcdu);
    ShaderResourceTable store_mip_rgba_rt = irt;
    store_mip_rgba_rt.resources.erase(
        std::remove_if(store_mip_rgba_rt.resources.begin(),
                       store_mip_rgba_rt.resources.end(),
                       [](const ShaderResource& resource) {
                           return resource.cls == ResourceClass::StorageImage;
                       }),
        store_mip_rgba_rt.resources.end());
    ShaderResource store_mip_rgba_image{};
    store_mip_rgba_image.cls = ResourceClass::StorageImage;
    store_mip_rgba_image.binding = 4; store_mip_rgba_image.fetch_pc = 6;
    store_mip_rgba_image.sgpr_base = 12; store_mip_rgba_image.img_dim = 1;
    store_mip_rgba_image.format = DataFormat::Uint8; store_mip_rgba_image.num_components = 4;
    store_mip_rgba_image.width = W;
    store_mip_rgba_image.height = store_mip_rgba_image.depth = 1;
    store_mip_rgba_image.gpu_addr =
        reinterpret_cast<uint64_t>(store_mip_rgba_dst.data());
    store_mip_rgba_image.size = static_cast<uint32_t>(store_mip_rgba_dst.size());
    store_mip_rgba_image.proven_zero_mip = true;
    store_mip_rgba_rt.resources.push_back(store_mip_rgba_image);
    const std::vector<uint32_t> gta_store_mip_rgba_spirv = recompile_valu(
        gta_store_mip_xyzw_2d, std::size(gta_store_mip_xyzw_2d), 1, 0,
        &store_mip_rgba_rt);
    CHECK(!gta_store_mip_rgba_spirv.empty(),
          "GTA V's live dmask-xyzw IMAGE_STORE_MIP packet recompiles exactly");
    if (!gta_store_mip_rgba_spirv.empty()) {
        ComputeItem item;
        item.spirv = gta_store_mip_rgba_spirv;
        item.resources = std::make_shared<ShaderResourceTable>(store_mip_rgba_rt);
        item.launch.threads_x = W; item.launch.local_x = 64; item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x2042f49800ull;
        CHECK(prosper::frontend::execute_live_compute_items({item}),
              "live backend executes 0x2042f49800's dmask-xyzw IMAGE_STORE_MIP");
        bool exact_rgba = true;
        for (uint32_t x = 0; x < W; ++x) {
            exact_rgba &= store_mip_rgba_dst[x * 4u + 0u] == static_cast<uint8_t>(x) &&
                          store_mip_rgba_dst[x * 4u + 1u] == 1u &&
                          store_mip_rgba_dst[x * 4u + 2u] == 2u &&
                          store_mip_rgba_dst[x * 4u + 3u] == 3u;
        }
        CHECK(exact_rgba,
              "dmask-xyzw IMAGE_STORE_MIP writes all four live Uint8 channels at mip zero");
    }

    // --- #590: the same production storage-image copy across the INTEGER and packed formats DOLL's
    // UE4 post-process writes its color-grading LUT / exposure / volume dispatches as (a 32x32x32
    // Uint8/Unorm2_10_10_10 3D LUT, a 1x1x1 exposure, a 16x16x16 Uint16 volume). Each dispatch was
    // skipped before storage_(un)pack gained these formats, leaving the tonemap sampling an
    // unproduced surface. A raw uvec4 copy through the production backend must round-trip the guest
    // bytes bit-exact: integer formats widen/truncate, and any quantized 10/10/10/2 word is stable
    // under unpack->pack.
    auto run_format_copy = [&](DataFormat fmt, uint32_t ncomp, uint32_t texel_bytes,
                               const std::vector<uint8_t>& src) -> std::vector<uint8_t> {
        std::vector<uint8_t> dst(src.size(), 0x5A);
        ShaderResourceTable frt;
        frt.resources = irt.resources;   // reuse constant buffers 0..3 (binding 0 -> v0 = gid)
        frt.resources.erase(std::remove_if(frt.resources.begin(), frt.resources.end(),
            [](const ShaderResource& r) { return r.cls == ResourceClass::StorageImage; }),
            frt.resources.end());
        auto add_fmt_image = [&](uint32_t binding, uint32_t sgpr, void* data) {
            ShaderResource im{};
            im.cls = ResourceClass::StorageImage; im.img_dim = 0; im.binding = binding;
            im.sgpr_base = sgpr; im.format = fmt; im.num_components = ncomp;
            im.width = W; im.height = 1;
            im.gpu_addr = (uint64_t)(uintptr_t)data; im.size = W * texel_bytes;
            frt.resources.push_back(im);
        };
        add_fmt_image(4, 0, const_cast<uint8_t*>(src.data()));
        add_fmt_image(5, 8, dst.data());
        std::vector<uint32_t> spv = recompile_valu(
            image_copy, sizeof(image_copy) / sizeof(image_copy[0]), 1, 0, &frt);
        if (spv.empty()) return {};
        ComputeItem it;
        it.spirv = spv;
        it.resources = std::make_shared<ShaderResourceTable>(frt);
        it.launch.threads_x = W; it.launch.local_x = 64; it.launch.groups_x = 1;
        it.launch.local_y = it.launch.local_z = 1; it.launch.groups_y = it.launch.groups_z = 1;
        it.code_addr = 0x590591;
        if (!prosper::frontend::execute_live_compute_items({it})) return {};
        return dst;
    };
    {   // Uint8 x4 (fmt=11): raw integer channels, exact round-trip
        std::vector<uint8_t> s(W * 4);
        for (uint32_t i = 0; i < s.size(); i++) s[i] = (uint8_t)(i * 53 + 7);
        CHECK(run_format_copy(DataFormat::Uint8, 4, 4, s) == s,
              "Uint8x4 storage copy round-trips guest bytes bit-exact (#590)");
    }
    {   // Unorm8 x2: native RG8 storage keeps its two-byte texel width and conversion contract
        std::vector<uint8_t> s(W * 2);
        for (uint32_t i = 0; i < s.size(); i++) s[i] = (uint8_t)(i * 71 + 11);
        CHECK(run_format_copy(DataFormat::Unorm8, 2, 2, s) == s,
              "Unorm8x2 storage copy round-trips native RG8 texels bit-exact (#590)");
    }
    {   // Astro Bot's title composite writes a tiled 256x64 RGBA16-UNORM surface.
        std::vector<uint8_t> s(W * 8);
        for (uint32_t t = 0; t < W * 4; ++t) {
            const uint16_t value = static_cast<uint16_t>(t * 251u + 17u);
            s[t * 2] = static_cast<uint8_t>(value);
            s[t * 2 + 1] = static_cast<uint8_t>(value >> 8);
        }
        CHECK(run_format_copy(DataFormat::Unorm16, 4, 8, s) == s,
              "Unorm16x4 storage copy round-trips guest channels bit-exact (#590)");
    }
    {   // Uint16 x1 (fmt=7): 16-bit integer widen on load, low-16-bit truncate on store
        std::vector<uint8_t> s(W * 2);
        for (uint32_t i = 0; i < s.size(); i++) s[i] = (uint8_t)(i * 29 + 3);
        CHECK(run_format_copy(DataFormat::Uint16, 1, 2, s) == s,
              "Uint16x1 storage copy round-trips guest bytes bit-exact (#590)");
    }
    {   // Unorm16 x1 (fmt=4): normalized conversion preserves every quantized source value
        std::vector<uint8_t> s(W * 2);
        for (uint32_t t = 0; t < W; ++t) {
            const uint16_t value = static_cast<uint16_t>(t * 1040u);
            std::memcpy(&s[t * 2], &value, sizeof(value));
        }
        CHECK(run_format_copy(DataFormat::Unorm16, 1, 2, s) == s,
              "Unorm16x1 storage copy round-trips normalized guest values");
    }
    {   // Snorm8 x1 (fmt=9): -128 and -127 both represent -1 and write back canonically as -127
        const int8_t values[] = {-128, -127, -96, -64, -1, 0, 1, 63, 96, 127};
        std::vector<uint8_t> s(W), expected(W);
        for (uint32_t t = 0; t < W; ++t) {
            const int8_t value = values[t % std::size(values)];
            s[t] = static_cast<uint8_t>(value);
            expected[t] = static_cast<uint8_t>(value == -128 ? -127 : value);
        }
        CHECK(run_format_copy(DataFormat::Snorm8, 1, 1, s) == expected,
              "Snorm8x1 storage copy normalizes and canonicalizes the negative endpoint");
    }
    {   // Snorm16 x1 (fmt=5): The Plucky Squire uses this storage surface in its title path
        const int16_t values[] = {-32768, -32767, -24576, -16384, -1, 0, 1, 16383, 24576, 32767};
        std::vector<uint8_t> s(W * 2), expected(W * 2);
        for (uint32_t t = 0; t < W; ++t) {
            const int16_t value = values[t % std::size(values)];
            const int16_t canonical = value == -32768 ? -32767 : value;
            std::memcpy(&s[t * 2], &value, sizeof(value));
            std::memcpy(&expected[t * 2], &canonical, sizeof(canonical));
        }
        CHECK(run_format_copy(DataFormat::Snorm16, 1, 2, s) == expected,
              "Snorm16x1 storage copy normalizes and canonicalizes the negative endpoint");
    }
    {   // Unorm2_10_10_10 x4 (fmt=21): packed 10/10/10/2, quantized word stable under unpack->pack
        std::vector<uint8_t> s(W * 4);
        for (uint32_t t = 0; t < W; t++) { uint32_t p = t * 2654435761u; std::memcpy(&s[t * 4], &p, 4); }
        CHECK(run_format_copy(DataFormat::Unorm2_10_10_10, 4, 4, s) == s,
              "Unorm2_10_10_10 storage copy round-trips packed guest word bit-exact (#590)");
    }
    {   // R11G11B10 UFLOAT: native float storage conversion must preserve quantized finite texels
        std::vector<uint8_t> s(W * 4);
        for (uint32_t t = 0; t < W; ++t) {
            const float r = static_cast<float>((t * 17u) % 97u) / 13.0f;
            const float g = static_cast<float>((t * 29u) % 89u) / 11.0f;
            const float b = static_cast<float>((t * 43u) % 83u) / 9.0f;
            const uint32_t packed = static_cast<uint32_t>(float_to_f11(r)) |
                                    (static_cast<uint32_t>(float_to_f11(g)) << 11) |
                                    (static_cast<uint32_t>(float_to_f10(b)) << 22);
            std::memcpy(&s[t * 4], &packed, sizeof(packed));
        }
        CHECK(run_format_copy(DataFormat::Float10_11_11, 3, 4, s) == s,
              "R11G11B10 storage copy round-trips native packed texels bit-exact (#590)");
    }
    {   // No native R11 storage: shader-side R32_UINT packing must preserve every f11 code.
        constexpr uint32_t PACKED_W = 2048;
        std::vector<uint32_t> indices(PACKED_W), src(PACKED_W), dst(PACKED_W, 0x5a5a5a5au);
        for (uint32_t t = 0; t < PACKED_W; ++t) {
            indices[t] = t;
            src[t] = t | (((t * 1031u) & 0x7ffu) << 11) | ((t & 0x3ffu) << 22);
        }
        ShaderResourceTable packed_rt = irt;
        for (ShaderResource& resource : packed_rt.resources) {
            if (resource.binding == 0) {
                resource.gpu_addr = reinterpret_cast<uint64_t>(indices.data());
                resource.size = static_cast<uint32_t>(indices.size() * sizeof(uint32_t));
            }
            if (resource.binding != 4 && resource.binding != 5) continue;
            resource.img_dim = 1;
            resource.format = DataFormat::Float10_11_11;
            resource.num_components = 3;
            resource.width = PACKED_W;
            resource.height = resource.depth = 1;
            resource.tile_mode = 0;
            resource.gpu_addr = reinterpret_cast<uint64_t>(
                resource.binding == 4 ? src.data() : dst.data());
            resource.size = PACKED_W * sizeof(uint32_t);
        }
        const std::vector<uint32_t> packed_spirv = recompile_valu(
            image_copy_2d, std::size(image_copy_2d), 1, 0, &packed_rt);
        const auto packed_report = validate_spirv_descriptor_interface(
            packed_spirv, &packed_rt, 0, SpirvShaderStage::Compute, false);
        CHECK(!packed_spirv.empty() && packed_report.ok() &&
                  packed_report.descriptors.size() == 4 &&
                  packed_report.descriptors[2].storage_image_format == kSpirvImageFormatR32ui &&
                  packed_report.descriptors[3].storage_image_format == kSpirvImageFormatR32ui,
              "R11G11B10 fallback recompiles both 2D storage views as typed R32ui");
        if (!packed_spirv.empty()) {
            ComputeItem packed_item;
            packed_item.spirv = packed_spirv;
            packed_item.resources = std::make_shared<ShaderResourceTable>(packed_rt);
            packed_item.launch.threads_x = PACKED_W;
            packed_item.launch.local_x = 64;
            packed_item.launch.groups_x = PACKED_W / 64;
            packed_item.launch.local_y = packed_item.launch.local_z = 1;
            packed_item.launch.groups_y = packed_item.launch.groups_z = 1;
            packed_item.code_addr = 0x590b10;
            const bool packed_executed =
                prosper::frontend::execute_live_compute_items({packed_item});

            // #1681. Two separate corrections live here.
            //
            // The INSTRUMENT: the old report used std::mismatch, which returns only the FIRST
            // differing element while the assertion compared the whole vector — so its printout was
            // byte-identical whether 1 texel differed or 92, and a breadth claim read off it had no
            // evidence behind it. The census below counts and classifies every differing texel.
            //
            // The CONTRACT: the sweep covers all 2048 f11 codes, which necessarily includes the NaN
            // encodings, and it required bit-exact equality over them. NaN *payload* propagation is
            // not something any driver owes us — IEEE 754 leaves it unspecified and SPIR-V/Vulkan
            // follow suit — and prosper's unpack routes each code through GLSL UnpackHalf2x16, a
            // genuine f16->f32 conversion. RADV preserves the payload; lavapipe quiets signalling
            // NaNs (x86 VCVTPH2PS sets the quiet bit), and both are conformant. So require
            // bit-exactness for every finite and infinite code — the values that actually carry
            // rendered colour — and require only that a NaN stays a NaN. This keeps all 2048 codes
            // under test and drops only the bits no implementation guarantees.
            struct Field { uint32_t shift, width; };
            const Field fields[3] = {{0, 11}, {11, 11}, {22, 10}};  // R f11, G f11, B f10
            auto chan = [&](uint32_t texel, int c) {
                return (texel >> fields[c].shift) & ((1u << fields[c].width) - 1u);
            };
            auto mant_bits = [&](int c) { return fields[c].width - 5u; };
            auto is_nan = [&](uint32_t v, int c) {
                const uint32_t m = mant_bits(c);
                return (v >> m) == 0x1fu && (v & ((1u << m) - 1u)) != 0u;
            };
            auto is_inf = [&](uint32_t v, int c) {
                const uint32_t m = mant_bits(c);
                return (v >> m) == 0x1fu && (v & ((1u << m) - 1u)) == 0u;
            };
            auto is_snan = [&](uint32_t v, int c) {
                const uint32_t m = mant_bits(c);
                return is_nan(v, c) && (v & (1u << (m - 1u))) == 0u;
            };
            // A channel round-trips when it is bit-exact, or when a NaN source stayed some NaN.
            auto channel_round_trips = [&](uint32_t s, uint32_t d, int c) {
                const uint32_t sv = chan(s, c), dv = chan(d, c);
                return is_nan(sv, c) ? is_nan(dv, c) : sv == dv;
            };

            // Pin the narrowed predicate itself on the CPU, with no GPU involved (#1681). The live
            // sweep can only demonstrate the cases the driver happens to produce, so it cannot show
            // that the exemption stops at NaN *payload* — and a narrowing that quietly also accepted
            // Inf<->NaN or NaN->finite would still pass every run. These assertions fix the contract
            // in both directions and would fail if a later edit widened it.
            {
                bool accepts_payload = true, rejects_everything_else = true;
                for (int c = 0; c < 3; ++c) {
                    const uint32_t m = mant_bits(c), sh = fields[c].shift;
                    auto tex = [&](uint32_t v) { return v << sh; };
                    const uint32_t inf = 0x1fu << m;                 // exp all ones, mantissa 0
                    const uint32_t snan = inf | 1u;                  // mantissa MSB clear
                    const uint32_t qnan = inf | (1u << (m - 1u));    // mantissa MSB set
                    const uint32_t sub = 1u;                         // exp 0, mantissa 1
                    const uint32_t norm = (15u << m) | 3u;           // ordinary finite value
                    // Payload movement between NaNs is permitted, in either direction.
                    accepts_payload &= channel_round_trips(tex(snan), tex(qnan), c);
                    accepts_payload &= channel_round_trips(tex(qnan), tex(snan), c);
                    accepts_payload &= channel_round_trips(tex(snan), tex(snan), c);
                    accepts_payload &= channel_round_trips(tex(inf), tex(inf), c);
                    accepts_payload &= channel_round_trips(tex(sub), tex(sub), c);
                    accepts_payload &= channel_round_trips(tex(norm), tex(norm), c);
                    // Nothing else is. A NaN may not decay, and a non-NaN may not become one.
                    rejects_everything_else &= !channel_round_trips(tex(snan), tex(inf), c);
                    rejects_everything_else &= !channel_round_trips(tex(qnan), tex(inf), c);
                    rejects_everything_else &= !channel_round_trips(tex(snan), tex(0), c);
                    rejects_everything_else &= !channel_round_trips(tex(snan), tex(norm), c);
                    rejects_everything_else &= !channel_round_trips(tex(inf), tex(snan), c);
                    rejects_everything_else &= !channel_round_trips(tex(inf), tex(norm), c);
                    rejects_everything_else &= !channel_round_trips(tex(norm), tex(norm + 1u), c);
                    rejects_everything_else &= !channel_round_trips(tex(sub), tex(0), c);
                    rejects_everything_else &= !channel_round_trips(tex(0), tex(sub), c);
                }
                CHECK(accepts_payload,
                      "R11G11B10 round-trip contract permits NaN payload movement in both directions");
                CHECK(rejects_everything_else,
                      "R11G11B10 round-trip contract still rejects NaN<->Inf, NaN->finite, and any "
                      "finite or subnormal change in every channel");
            }

            size_t payload_only = 0, real_diffs = 0;
            for (size_t t = 0; t < src.size(); ++t) {
                if (src[t] == dst[t]) continue;
                bool ok = true;
                for (int c = 0; c < 3; ++c)
                    if (!channel_round_trips(src[t], dst[t], c)) ok = false;
                if (ok) ++payload_only; else ++real_diffs;
            }
            if (packed_executed && dst != src) {
                size_t total = 0, all_finite = 0, any_snan = 0, any_nan = 0, any_inf = 0;
                size_t quiet_exact = 0, canon_exact = 0;
                long long first_finite = -1, first_any = -1;
                for (size_t t = 0; t < src.size(); ++t) {
                    if (src[t] == dst[t]) continue;
                    ++total;
                    if (first_any < 0) first_any = static_cast<long long>(t);
                    bool finite = true, snan = false, nan = false, inf = false;
                    uint32_t quieted = 0, canonical = 0;
                    for (int c = 0; c < 3; ++c) {
                        const uint32_t v = chan(src[t], c), m = mant_bits(c);
                        if (is_nan(v, c)) { nan = true; finite = false; }
                        if (is_inf(v, c)) { inf = true; finite = false; }
                        if (is_snan(v, c)) snan = true;
                        // Quieting model: an sNaN gains the mantissa MSB; everything else is kept.
                        const uint32_t q = is_snan(v, c) ? (v | (1u << (m - 1u))) : v;
                        // Canonicalisation model: every NaN collapses to exp=31, mantissa MSB only.
                        const uint32_t k = is_nan(v, c) ? ((0x1fu << m) | (1u << (m - 1u))) : v;
                        quieted |= q << fields[c].shift;
                        canonical |= k << fields[c].shift;
                    }
                    if (finite) {
                        ++all_finite;
                        if (first_finite < 0) first_finite = static_cast<long long>(t);
                    }
                    if (snan) ++any_snan;
                    if (nan) ++any_nan;
                    if (inf && !nan) ++any_inf;
                    if (dst[t] == quieted) ++quiet_exact;
                    if (dst[t] == canonical) ++canon_exact;
                    if (total <= 8 || finite)
                        std::printf("  packed R11 diff texel=%zu src=%08x dst=%08x xor=%08x%s\n",
                                    t, src[t], dst[t], src[t] ^ dst[t],
                                    finite ? "  ALL-FINITE" : "");
                }
                std::printf("  packed R11 census: total=%zu all-finite=%zu any-sNaN=%zu "
                            "any-NaN=%zu inf-only=%zu quiet-model-exact=%zu canon-model-exact=%zu "
                            "first=%lld first-finite=%lld nan-payload-only=%zu real=%zu\n",
                            total, all_finite, any_snan, any_nan, any_inf, quiet_exact,
                            canon_exact, first_any, first_finite, payload_only, real_diffs);
            }
            CHECK(packed_executed && real_diffs == 0,
                  "shader-side R11G11B10 pack/unpack round-trips every finite and infinite f11 "
                  "code exactly, and preserves NaN-ness (payload is implementation-defined)");
        }

        // Also compare arbitrary raw f32 channel bits against the CPU's exact RNE oracle. The
        // representable-code round-trip above cannot exercise values on either side of a packing
        // boundary, negative clamping, f32 underflow, overflow, or payload truncation.
        std::vector<uint32_t> float_channels(PACKED_W * 4), expected(PACKED_W);
        const uint32_t edge_bits[] = {
            0x00000000u, 0x00000001u, 0x007fffffu, 0x00800000u,
            0x80000000u, 0xbf800000u, 0x3f800000u, 0x3f810000u,
            0x477fe000u, 0x477ff000u, 0x47800000u, 0x7f800000u,
            0xff800000u, 0x7f800001u, 0x7fc12345u, 0xffc54321u,
        };
        auto bits_float = [](uint32_t bits) {
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        };
        for (uint32_t t = 0; t < PACKED_W; ++t) {
            for (uint32_t c = 0; c < 4; ++c)
                float_channels[t * 4 + c] = t < std::size(edge_bits)
                    ? edge_bits[(t + c * 5u) % std::size(edge_bits)]
                    : (t * 2654435761u + c * 2246822519u);
            expected[t] = static_cast<uint32_t>(float_to_f11(
                              bits_float(float_channels[t * 4 + 0]))) |
                          (static_cast<uint32_t>(float_to_f11(
                              bits_float(float_channels[t * 4 + 1]))) << 11) |
                          (static_cast<uint32_t>(float_to_f10(
                              bits_float(float_channels[t * 4 + 2]))) << 22);
        }
        std::fill(dst.begin(), dst.end(), 0x5a5a5a5au);
        ShaderResourceTable conversion_rt = packed_rt;
        for (ShaderResource& resource : conversion_rt.resources) {
            if (resource.binding != 4 && resource.binding != 5) continue;
            resource.gpu_addr = reinterpret_cast<uint64_t>(
                resource.binding == 4 ? float_channels.data() : dst.data());
            if (resource.binding == 4) {
                resource.format = DataFormat::Float32;
                resource.num_components = 4;
                resource.size = static_cast<uint32_t>(float_channels.size() * sizeof(uint32_t));
            }
        }
        const std::vector<uint32_t> conversion_spirv = recompile_valu(
            image_copy_2d, std::size(image_copy_2d), 1, 0, &conversion_rt);
        if (!conversion_spirv.empty()) {
            ComputeItem conversion_item;
            conversion_item.spirv = conversion_spirv;
            conversion_item.resources = std::make_shared<ShaderResourceTable>(conversion_rt);
            conversion_item.launch.threads_x = PACKED_W;
            conversion_item.launch.local_x = 64;
            conversion_item.launch.groups_x = PACKED_W / 64;
            conversion_item.launch.local_y = conversion_item.launch.local_z = 1;
            conversion_item.launch.groups_y = conversion_item.launch.groups_z = 1;
            conversion_item.code_addr = 0x590b11;
            CHECK(prosper::frontend::execute_live_compute_items({conversion_item}) &&
                      dst == expected,
                  "shader-side R11G11B10 packing matches the CPU oracle for arbitrary f32 bits");
        } else {
            CHECK(false, "mixed Float32-to-R11G11B10 storage conversion recompiles");
        }
    }

    // The backend publishes ordinary tiled texels, not hardware-compressed blocks. Prove that a
    // DCC-enabled storage destination atomically becomes the uncompressed (0xff) metadata state,
    // including replay-owned metadata that a later sampled descriptor shares by logical address.
    const uint32_t dcc_tile = static_cast<uint32_t>(TileMode::Sw64KbRX);

    // The production sampled-image path must agree with a semantic uncompressed reference, not with
    // the old compressed-base fallback. A zero base plus DCC_CLEAR_0001 logically contains
    // FP16 (0,0,0,1); copy both representations through the same generated shader and require the
    // resulting guest RGBA8 storage bytes and hashes to match exactly.
    {
        const size_t clear_source_bytes = tiled_surface_bytes(W, 1, dcc_tile, 0, 8);
        const size_t clear_metadata_bytes =
            gfx10_dcc_metadata_bytes(W, 1, 1, dcc_tile, 8, true);
        std::vector<uint8_t> compressed_base(clear_source_bytes, 0);
        std::vector<uint8_t> clear_metadata(clear_metadata_bytes, 0x40);
        std::vector<uint8_t> reference_linear(W * 8, 0);
        const uint16_t one = float_to_half(1.0f);
        for (uint32_t texel = 0; texel < W; ++texel)
            std::memcpy(reference_linear.data() + texel * 8 + 6, &one, sizeof(one));
        std::vector<uint8_t> reference_tiled(clear_source_bytes, 0);
        tile_surface(reference_tiled.data(), reference_linear.data(), W, 1,
                     dcc_tile, 0, 8);
        std::vector<uint8_t> clear_dst(W * 4, 0xa5);
        std::vector<uint8_t> reference_dst(W * 4, 0x5a);

        auto sampled_clear_table = [&](std::vector<uint8_t>& source,
                                       std::vector<uint8_t>& destination,
                                       bool compressed) {
            ShaderResourceTable table = irt;
            for (ShaderResource& resource : table.resources) {
                if (resource.binding != 4 && resource.binding != 5) continue;
                resource.img_dim = 1;
                resource.width = W;
                resource.height = resource.depth = 1;
                resource.declared_mip_levels = 1;
                resource.in_mip_tail = false;
                resource.layer_stride_bytes = resource.layer_mip_offset_bytes = 0;
                if (resource.binding == 4) {
                    resource.cls = ResourceClass::Texture;
                    resource.format = DataFormat::Float16;
                    resource.num_components = 4;
                    resource.tile_mode = dcc_tile;
                    resource.gpu_addr = reinterpret_cast<uint64_t>(source.data());
                    resource.size = static_cast<uint32_t>(source.size());
                    resource.swizzle[0] = 4;
                    resource.swizzle[1] = 5;
                    resource.swizzle[2] = 6;
                    resource.swizzle[3] = 7;
                    resource.compression_enabled = compressed;
                    resource.meta_pipe_aligned = compressed;
                    resource.alpha_is_on_msb = compressed;
                    resource.metadata_addr = compressed ? 0x301758d000ull : 0;
                    resource.dcc_metadata_size = compressed ? clear_metadata.size() : 0;
                    resource.dcc_metadata_host_data = compressed ? clear_metadata.data() : nullptr;
                    resource.dcc_metadata_host_data_size = compressed ? clear_metadata.size() : 0;
                } else {
                    resource.cls = ResourceClass::StorageImage;
                    resource.format = DataFormat::Unorm8;
                    resource.num_components = 4;
                    resource.tile_mode = 0;
                    resource.gpu_addr = reinterpret_cast<uint64_t>(destination.data());
                    resource.size = static_cast<uint32_t>(destination.size());
                }
            }
            return table;
        };
        ShaderResourceTable clear_rt = sampled_clear_table(
            compressed_base, clear_dst, true);
        ShaderResourceTable reference_rt = sampled_clear_table(
            reference_tiled, reference_dst, false);
        const std::vector<uint32_t> clear_spirv = recompile_valu(
            image_copy_2d, std::size(image_copy_2d), 1, 0, &clear_rt);
        CHECK(!clear_spirv.empty(),
              "sampled DCC fast-clear semantic-reference kernel recompiles");
        auto run_clear = [&](ShaderResourceTable& table, uint64_t code_addr) {
            if (clear_spirv.empty()) return false;
            ComputeItem item;
            item.spirv = clear_spirv;
            item.resources = std::make_shared<ShaderResourceTable>(table);
            item.launch.threads_x = W;
            item.launch.local_x = 64;
            item.launch.groups_x = 1;
            item.launch.local_y = item.launch.local_z = 1;
            item.launch.groups_y = item.launch.groups_z = 1;
            item.code_addr = code_addr;
            return prosper::frontend::execute_live_compute_items({item});
        };
        const bool clear_ok = run_clear(clear_rt, 0x301758d1) &&
                              run_clear(reference_rt, 0x301758d2);
        std::vector<uint8_t> expected(W * 4, 0);
        for (uint32_t texel = 0; texel < W; ++texel) expected[texel * 4 + 3] = 255;
        const uint64_t clear_hash = gpu_capture_hash(clear_dst.data(), clear_dst.size());
        const uint64_t reference_hash =
            gpu_capture_hash(reference_dst.data(), reference_dst.size());
        const uint64_t expected_hash = gpu_capture_hash(expected.data(), expected.size());
        CHECK(clear_ok && clear_dst == reference_dst && clear_dst == expected &&
                  clear_hash == reference_hash && clear_hash == expected_hash,
              "sampled DCC clear output matches explicit uncompressed FP16 (0,0,0,1)");
    }

    const size_t tiled_bytes = tiled_surface_bytes(W, 1, dcc_tile, 0, 4);
    const size_t metadata_bytes = gfx10_dcc_metadata_bytes(W, 1, 1, dcc_tile, 4, true);
    std::vector<uint8_t> tiled_src(tiled_bytes, 0), tiled_dst(tiled_bytes, 0);
    std::vector<uint8_t> dcc_metadata(metadata_bytes, 0x40);
    tile_surface(tiled_src.data(), img_src.data(), W, 1, dcc_tile, 0, 4);
    ShaderResourceTable dcc_rt = irt;
    for (ShaderResource& resource : dcc_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 1;
        resource.tile_mode = dcc_tile;
        resource.size = static_cast<uint32_t>(tiled_bytes);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? tiled_src.data() : tiled_dst.data());
        if (resource.binding == 5) {
            resource.compression_enabled = true;
            resource.write_compress_enabled = true;
            resource.meta_pipe_aligned = true;
            resource.metadata_addr = 0x207cef0000ull;
            resource.dcc_metadata_size = metadata_bytes;
            resource.dcc_metadata_host_data = dcc_metadata.data();
            resource.dcc_metadata_host_data_size = dcc_metadata.size();
        }
    }
    ComputeShaderConfig dcc_config;
    dcc_config.user_sgprs.resize(16);
    dcc_config.local_x = W;
    dcc_config.local_y = dcc_config.local_z = 1;
    dcc_config.threads_x = W;
    dcc_config.threads_y = dcc_config.threads_z = 1;
    dcc_config.native_storage_format_support =
        native_storage_format_support_bit(DataFormat::Unorm8, 4);
    std::vector<uint32_t> dcc_spirv = recompile_compute(
        image_copy_2d, std::size(image_copy_2d), &dcc_rt, dcc_config);
    const DescriptorValidationReport dcc_report = validate_spirv_descriptor_interface(
        dcc_spirv, &dcc_rt, 0, SpirvShaderStage::Compute, false);
    const SpirvDescriptorBinding* dcc_output_binding =
        find_spirv_descriptor_binding(dcc_report, 0, 5);
    const bool dcc_shape_ok = !dcc_spirv.empty() && dcc_report.ok() &&
        dcc_output_binding && dcc_output_binding->storage_float &&
        dcc_output_binding->writable;
    CHECK(dcc_shape_ok,
          "DCC producer recompiles to exact typed writable storage for sampled export");
    ComputeItem dcc_item;
    dcc_item.spirv = dcc_spirv;
    dcc_item.resources = std::make_shared<ShaderResourceTable>(dcc_rt);
    dcc_item.launch.threads_x = W;
    dcc_item.launch.threads_y = dcc_item.launch.threads_z = 1;
    dcc_item.launch.local_x = 64;
    dcc_item.launch.groups_x = 1;
    dcc_item.launch.local_y = dcc_item.launch.local_z = 1;
    dcc_item.launch.groups_y = dcc_item.launch.groups_z = 1;
    dcc_item.code_addr = 0x719dcc;
    if (dcc_shape_ok) {
        // The first sighting of a write-only storage kernel is deliberately a poison proving
        // frame.  Promotion must stay fail-closed until that execution proves full coverage; then
        // restore a compressed metadata state and require the proven seed-skip execution to publish
        // the exact post-writeback image.
        const uint64_t promotions_before =
            prosper::frontend::live_compute_dcc_post_writeback_promotions();
        CHECK(prosper::frontend::execute_live_compute_items({dcc_item}),
              "live backend proves coverage while writing the tiled DCC storage image");
        CHECK(prosper::frontend::live_compute_dcc_post_writeback_promotions() ==
                  promotions_before,
              "poison-proving DCC writeback publishes no cache authority");
        std::fill(dcc_metadata.begin(), dcc_metadata.end(), 0x40);
        CHECK(prosper::frontend::execute_live_compute_items({dcc_item}),
              "proven full-coverage backend rewrites the tiled DCC storage image");
        CHECK(prosper::frontend::live_compute_dcc_post_writeback_promotions() ==
                  promotions_before + 1,
              "successful exact DCC writeback promotes its transient storage image");
        std::vector<uint8_t> dcc_linear(W * 4, 0);
        detile_surface(dcc_linear.data(), tiled_dst.data(), W, 1, dcc_tile, 0, 4);
        CHECK(dcc_linear == img_src,
              "DCC storage writeback preserves the producer's tiled base texels");
        CHECK(std::all_of(dcc_metadata.begin(), dcc_metadata.end(),
                          [](uint8_t code) { return code == 0xff; }),
              "DCC storage writeback publishes uniform uncompressed metadata");

        // Consume the just-published target through an ordinary sampled descriptor.  Guest bytes
        // are the correctness oracle; the monotonic seed counter independently proves that the
        // consumer borrowed the producer's retained Vulkan image instead of uploading those bytes.
        std::vector<uint8_t> dcc_copy_dst(tiled_bytes, 0x6d);
        ShaderResourceTable dcc_consumer_rt = irt;
        const ShaderResource* dcc_output = nullptr;
        for (const ShaderResource& resource : dcc_rt.resources)
            if (resource.binding == 5) dcc_output = &resource;
        for (ShaderResource& resource : dcc_consumer_rt.resources) {
            if (!dcc_output || (resource.binding != 4 && resource.binding != 5)) continue;
            const uint32_t binding = resource.binding;
            const uint32_t sgpr_base = resource.sgpr_base;
            resource = *dcc_output;
            resource.binding = binding;
            resource.sgpr_base = sgpr_base;
            if (binding == 4) {
                resource.cls = ResourceClass::Texture;
                resource.gpu_addr = reinterpret_cast<uint64_t>(tiled_dst.data());
                resource.swizzle[0] = 4;
                resource.swizzle[1] = 5;
                resource.swizzle[2] = 6;
                resource.swizzle[3] = 7;
            } else {
                resource.cls = ResourceClass::StorageImage;
                resource.gpu_addr = reinterpret_cast<uint64_t>(dcc_copy_dst.data());
                resource.compression_enabled = false;
                resource.write_compress_enabled = false;
                resource.meta_pipe_aligned = false;
                resource.metadata_addr = 0;
                resource.dcc_metadata_size = 0;
                resource.dcc_metadata_host_data = nullptr;
                resource.dcc_metadata_host_data_size = 0;
            }
        }
        const std::vector<uint32_t> dcc_consumer_spirv = recompile_valu(
            image_copy_2d, std::size(image_copy_2d), 1, 0, &dcc_consumer_rt);
        CHECK(!dcc_consumer_spirv.empty(),
              "post-DCC sampled consumer recompiles against the exact producer descriptor");
        ComputeItem dcc_consumer_item = dcc_item;
        dcc_consumer_item.spirv = dcc_consumer_spirv;
        dcc_consumer_item.resources =
            std::make_shared<ShaderResourceTable>(dcc_consumer_rt);
        dcc_consumer_item.code_addr = 0x719dce;
        ShaderResource sampled_dcc_output = dcc_output ? *dcc_output : ShaderResource{};
        sampled_dcc_output.cls = ResourceClass::Texture;
        const auto run_dcc_consumer = [&]() {
            std::fill(dcc_copy_dst.begin(), dcc_copy_dst.end(), 0x6d);
            const uint64_t seeds_before =
                prosper::frontend::live_compute_storage_transfer_seeds();
            const bool executed = prosper::frontend::execute_live_compute_items(
                {dcc_consumer_item});
            const uint64_t seeds_after =
                prosper::frontend::live_compute_storage_transfer_seeds();
            std::vector<uint8_t> copied_linear(W * 4, 0);
            detile_surface(copied_linear.data(), dcc_copy_dst.data(),
                           W, 1, dcc_tile, 0, 4);
            return std::tuple{executed, copied_linear, seeds_before, seeds_after};
        };
        const auto run_dcc_ordered_handoff =
            [&](prosper::frontend::LiveComputeImageImport* graphics_import) {
                std::fill(dcc_copy_dst.begin(), dcc_copy_dst.end(), 0x6d);
                ComputeItem ordered_producer = dcc_item;
                ordered_producer.dispatch_index = 701;
                ordered_producer.command_order = 10;
                ComputeItem ordered_consumer = dcc_consumer_item;
                ordered_consumer.dispatch_index = 702;
                ordered_consumer.command_order = 20;
                DrawItem ordered_draw;
                ordered_draw.draw_index = 703;
                ordered_draw.command_order = 30;
                const uint64_t promotions_before =
                    prosper::frontend::live_compute_dcc_post_writeback_promotions();
                const uint64_t seeds_before =
                    prosper::frontend::live_compute_storage_transfer_seeds();
                const OrderedSubmitResult ordered = execute_ordered_items(
                    {{SubmitOperationKind::Dispatch, ordered_producer.dispatch_index,
                      ordered_producer.command_order},
                     {SubmitOperationKind::Dispatch, ordered_consumer.dispatch_index,
                      ordered_consumer.command_order},
                     {SubmitOperationKind::Draw, ordered_draw.draw_index,
                      ordered_draw.command_order}},
                    {ordered_draw}, {ordered_producer, ordered_consumer},
                    [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                        if (graphics_import)
                            (void)prosper::frontend::import_live_compute_storage_image(
                                sampled_dcc_output, tiled_bytes, *graphics_import);
                        return RenderedFrame{};
                    },
                    [&](const std::vector<ComputeItem>& items) {
                        return prosper::frontend::execute_live_compute_items(items);
                    },
                    1, 1);
                const uint64_t promotions_after =
                    prosper::frontend::live_compute_dcc_post_writeback_promotions();
                const uint64_t seeds_after =
                    prosper::frontend::live_compute_storage_transfer_seeds();
                std::vector<uint8_t> copied_linear(W * 4, 0);
                detile_surface(copied_linear.data(), dcc_copy_dst.data(),
                               W, 1, dcc_tile, 0, 4);
                return std::tuple{ordered.compute_executed, copied_linear,
                                  promotions_before, promotions_after,
                                  seeds_before, seeds_after};
            };
        if (!dcc_consumer_spirv.empty()) {
            std::fill(dcc_metadata.begin(), dcc_metadata.end(), 0x40);
            prosper::frontend::LiveComputeImageImport pinned_import;
            auto [consumer_ok, copied_linear, ordered_promotions_before,
                  ordered_promotions_after, seeds_before, seeds_after] =
                run_dcc_ordered_handoff(&pinned_import);
            CHECK(consumer_ok && copied_linear == img_src &&
                      ordered_promotions_after == ordered_promotions_before + 1,
                  "post-DCC sampled consumer receives every exact producer byte");
            CHECK(!native_2d_compute_transfer_available || seeds_after > seeds_before,
                  "post-DCC producer seeds its sampled consumer on-GPU");
            CHECK(native_2d_compute_transfer_available || seeds_after == seeds_before,
                  "disabled native transfer keeps the post-DCC sampled guest fallback");

            CHECK(pinned_import.valid(),
                  "post-DCC graphics import pins the exact promoted cache entry");
            if (pinned_import.valid()) {
                std::fill(dcc_metadata.begin(), dcc_metadata.end(), 0x40);
                for (size_t index = 0; index < img_src.size(); ++index)
                    img_src[index] = static_cast<uint8_t>(index * 29u + 11u);
                tile_surface(tiled_src.data(), img_src.data(), W, 1, dcc_tile, 0, 4);
                const uint64_t pinned_promotions_before =
                    prosper::frontend::live_compute_dcc_post_writeback_promotions();
                const uint64_t pinned_reuses_before =
                    prosper::frontend::live_compute_dcc_forced_seed_allocation_reuses();
                const uint64_t pinned_replacements_before =
                    prosper::frontend::live_compute_dcc_post_writeback_replacements();
                CHECK(prosper::frontend::execute_live_compute_items({dcc_item}),
                      "compressed producer completes while its old exact cache entry is pinned");
                CHECK(prosper::frontend::live_compute_dcc_post_writeback_promotions() ==
                          pinned_promotions_before &&
                          prosper::frontend::live_compute_dcc_forced_seed_allocation_reuses() ==
                              pinned_reuses_before &&
                          prosper::frontend::live_compute_dcc_post_writeback_replacements() ==
                              pinned_replacements_before,
                      "pinned exact cache entry declines reuse and post-DCC replacement");
                pinned_import = {};
                auto [pinned_consumer_ok, pinned_linear,
                      pinned_seeds_before, pinned_seeds_after] = run_dcc_consumer();
                CHECK(pinned_consumer_ok && pinned_linear == img_src,
                      "declined pinned promotion preserves the exact guest fallback");
                CHECK(pinned_seeds_after == pinned_seeds_before,
                      "declined pinned promotion publishes no new transfer authority");
            }

            // Repeat with the exact cache entry unpinned and a changed producer result. This is the
            // Astro frame shape: an older consumer entry exists under the same key, and compressed
            // guest bytes must be uploaded into its allocation without inheriting source authority.
            std::fill(dcc_metadata.begin(), dcc_metadata.end(), 0x40);
            for (size_t index = 0; index < img_src.size(); ++index)
                img_src[index] = static_cast<uint8_t>(index * 13u + 0x51u);
            tile_surface(tiled_src.data(), img_src.data(), W, 1, dcc_tile, 0, 4);
            const uint64_t replacement_promotions_before =
                prosper::frontend::live_compute_dcc_post_writeback_promotions();
            const uint64_t allocation_reuses_before =
                prosper::frontend::live_compute_dcc_forced_seed_allocation_reuses();
            const uint64_t replacements_before =
                prosper::frontend::live_compute_dcc_post_writeback_replacements();
            auto [replacement_ok, replacement_linear,
                  ordered_replacement_promotions_before,
                  ordered_replacement_promotions_after,
                  replacement_seeds_before, replacement_seeds_after] =
                run_dcc_ordered_handoff(nullptr);
            CHECK(replacement_ok &&
                      replacement_promotions_before ==
                          ordered_replacement_promotions_before &&
                      ordered_replacement_promotions_after ==
                          replacement_promotions_before + 1 &&
                      prosper::frontend::live_compute_dcc_forced_seed_allocation_reuses() ==
                          allocation_reuses_before + 1 &&
                      prosper::frontend::live_compute_dcc_post_writeback_replacements() ==
                          replacements_before,
                  "unpinned exact-key allocation is reused after a mandatory producer upload");
            CHECK(replacement_ok && replacement_linear == img_src,
                  "forced upload into a reused allocation observes changed producer bytes exactly");
            CHECK(!native_2d_compute_transfer_available ||
                      replacement_seeds_after > replacement_seeds_before,
                  "reused-allocation promotion restores producer-to-consumer image reuse");

            ShaderResource mismatched_dcc_output = sampled_dcc_output;
            ++mismatched_dcc_output.width;
            prosper::frontend::LiveComputeImageImport mismatched_import;
            CHECK(!prosper::frontend::import_live_compute_storage_image(
                      mismatched_dcc_output, tiled_bytes, mismatched_import),
                  "post-DCC graphics reuse requires the exact promoted cache key");

            // Mutate the mechanism, not the expected output: disable one eligible reuse while
            // leaving every cache-promotion gate intact. The changed bytes must remain exact, and
            // the independent counters must prove that the old replacement path actually ran.
            std::fill(dcc_metadata.begin(), dcc_metadata.end(), 0x40);
            for (size_t index = 0; index < img_src.size(); ++index)
                img_src[index] = static_cast<uint8_t>(index * 17u + 0x35u);
            tile_surface(tiled_src.data(), img_src.data(), W, 1, dcc_tile, 0, 4);
            prosper::frontend::live_compute_disable_next_dcc_allocation_reuse_for_test();
            const uint64_t mutated_reuses_before =
                prosper::frontend::live_compute_dcc_forced_seed_allocation_reuses();
            const uint64_t mutated_replacements_before =
                prosper::frontend::live_compute_dcc_post_writeback_replacements();
            auto [mutated_ok, mutated_linear, mutated_promotions_before,
                  mutated_promotions_after, mutated_seeds_before, mutated_seeds_after] =
                run_dcc_ordered_handoff(nullptr);
            CHECK(mutated_ok && mutated_linear == img_src &&
                      mutated_promotions_after == mutated_promotions_before + 1 &&
                      prosper::frontend::live_compute_dcc_forced_seed_allocation_reuses() ==
                          mutated_reuses_before &&
                      prosper::frontend::live_compute_dcc_post_writeback_replacements() ==
                          mutated_replacements_before + 1,
                  "disabling allocation reuse restores exact post-writeback replacement");
            CHECK(!native_2d_compute_transfer_available ||
                      mutated_seeds_after > mutated_seeds_before,
                  "replacement mutation preserves producer-to-consumer image reuse");

            // Force the real replacement cache-limit preflight below this allocation after exact
            // writeback and DCC publication. It must leave this dispatch transient, so the next
            // sampled use reads the correct guest fallback without inheriting the older exact-key
            // entry's transfer authority.
            std::fill(dcc_metadata.begin(), dcc_metadata.end(), 0x40);
            for (size_t index = 0; index < img_src.size(); ++index)
                img_src[index] = static_cast<uint8_t>(index * 7u + 0x23u);
            tile_surface(tiled_src.data(), img_src.data(), W, 1, dcc_tile, 0, 4);
            prosper::frontend::live_compute_disable_next_dcc_allocation_reuse_for_test();
            prosper::frontend::live_compute_limit_next_image_replacement_for_test();
            const uint64_t refused_promotions_before =
                prosper::frontend::live_compute_dcc_post_writeback_promotions();
            const uint64_t refused_reuses_before =
                prosper::frontend::live_compute_dcc_forced_seed_allocation_reuses();
            const uint64_t refused_replacements_before =
                prosper::frontend::live_compute_dcc_post_writeback_replacements();
            CHECK(prosper::frontend::execute_live_compute_items({dcc_item}) &&
                      prosper::frontend::live_compute_dcc_post_writeback_promotions() ==
                          refused_promotions_before &&
                      prosper::frontend::live_compute_dcc_forced_seed_allocation_reuses() ==
                          refused_reuses_before &&
                      prosper::frontend::live_compute_dcc_post_writeback_replacements() ==
                          refused_replacements_before,
                  "post-DCC capacity preflight preserves the transient fallback");
            auto [refused_ok, refused_linear,
                  refused_seeds_before, refused_seeds_after] = run_dcc_consumer();
            CHECK(refused_ok && refused_linear == img_src,
                  "declined admission preserves exact producer bytes through guest memory");
            CHECK(refused_seeds_after == refused_seeds_before,
                  "declined admission publishes no producer transfer authority");

            // The final metadata scan is an independent promotion gate.  Model an unresolved DCC
            // plane after otherwise successful data writeback and require both correct guest bytes
            // and absence of cache/transfer publication.
            std::fill(dcc_metadata.begin(), dcc_metadata.end(), 0x40);
            for (size_t index = 0; index < img_src.size(); ++index)
                img_src[index] = static_cast<uint8_t>(index * 43u + 3u);
            tile_surface(tiled_src.data(), img_src.data(), W, 1, dcc_tile, 0, 4);
            prosper::frontend::live_compute_leave_next_dcc_metadata_compressed_for_test();
            const uint64_t unresolved_promotions_before =
                prosper::frontend::live_compute_dcc_post_writeback_promotions();
            const uint64_t unresolved_reuses_before =
                prosper::frontend::live_compute_dcc_forced_seed_allocation_reuses();
            CHECK(prosper::frontend::execute_live_compute_items({dcc_item}) &&
                      prosper::frontend::live_compute_dcc_post_writeback_promotions() ==
                          unresolved_promotions_before &&
                      prosper::frontend::live_compute_dcc_forced_seed_allocation_reuses() ==
                          unresolved_reuses_before + 1 &&
                      std::any_of(dcc_metadata.begin(), dcc_metadata.end(),
                                  [](uint8_t code) { return code != 0xff; }),
                  "unresolved post-writeback DCC metadata invalidates the reused allocation");
            auto [unresolved_ok, unresolved_linear,
                  unresolved_seeds_before, unresolved_seeds_after] = run_dcc_consumer();
            CHECK(unresolved_ok && unresolved_linear == img_src,
                  "unresolved DCC promotion keeps the ordinary sampled guest fallback correct");
            CHECK(unresolved_seeds_after == unresolved_seeds_before,
                  "unresolved DCC metadata publishes no transfer authority");
        }

        // A failed readback on a fresh compressed target reaches neither base-byte publication nor
        // DCC promotion/export.  Use a fresh identity so an older valid cache entry cannot satisfy
        // the negative import control.
        std::vector<uint8_t> failed_dcc_dst(tiled_bytes, 0x77);
        const std::vector<uint8_t> failed_dcc_before = failed_dcc_dst;
        std::vector<uint8_t> failed_dcc_metadata(metadata_bytes, 0x40);
        ShaderResourceTable failed_dcc_rt = dcc_rt;
        ShaderResource failed_sampled_output;
        for (ShaderResource& resource : failed_dcc_rt.resources) {
            if (resource.binding != 5) continue;
            resource.gpu_addr = reinterpret_cast<uint64_t>(failed_dcc_dst.data());
            resource.metadata_addr = 0x207cef8000ull;
            resource.dcc_metadata_host_data = failed_dcc_metadata.data();
            resource.dcc_metadata_host_data_size = failed_dcc_metadata.size();
            failed_sampled_output = resource;
            failed_sampled_output.cls = ResourceClass::Texture;
        }
        ComputeItem failed_dcc_item = dcc_item;
        failed_dcc_item.resources =
            std::make_shared<ShaderResourceTable>(failed_dcc_rt);
        failed_dcc_item.code_addr = 0x719dcf;
        prosper::frontend::live_compute_fail_next_storage_readback_for_test();
        const uint64_t failed_promotions_before =
            prosper::frontend::live_compute_dcc_post_writeback_promotions();
        CHECK(!prosper::frontend::execute_live_compute_items({failed_dcc_item}) &&
                  prosper::frontend::live_compute_dcc_post_writeback_promotions() ==
                      failed_promotions_before &&
                  failed_dcc_dst == failed_dcc_before &&
                  std::all_of(failed_dcc_metadata.begin(), failed_dcc_metadata.end(),
                              [](uint8_t code) { return code == 0x40; }),
              "failed DCC readback publishes neither guest bytes, metadata, nor promotion");
        prosper::frontend::LiveComputeImageImport failed_import;
        CHECK(!prosper::frontend::import_live_compute_storage_image(
                  failed_sampled_output, tiled_bytes, failed_import),
              "failed DCC readback publishes no graphics cache authority");
    }
    // Two storage views may share base texels while only one carries DCC state. They are not safe
    // Vulkan-image aliases: collapsing the compressed view onto an uncompressed owner drops its
    // separate metadata writeback obligation and leaves later sampled users reading stale DCC.
    std::vector<uint8_t> mixed_alias_base = tiled_src;
    std::vector<uint8_t> mixed_alias_metadata(metadata_bytes, 0x40);
    ShaderResourceTable mixed_alias_rt = dcc_rt;
    for (ShaderResource& resource : mixed_alias_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.gpu_addr = reinterpret_cast<uint64_t>(mixed_alias_base.data());
        if (resource.binding == 5) {
            resource.metadata_addr = 0x207cf00000ull;
            resource.dcc_metadata_host_data = mixed_alias_metadata.data();
            resource.dcc_metadata_host_data_size = mixed_alias_metadata.size();
        }
    }
    const std::vector<uint32_t> mixed_alias_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &mixed_alias_rt);
    CHECK(!mixed_alias_spirv.empty(), "mixed DCC storage-alias kernel recompiles");
    if (!mixed_alias_spirv.empty()) {
        ComputeItem mixed_alias_item;
        mixed_alias_item.spirv = mixed_alias_spirv;
        mixed_alias_item.resources = std::make_shared<ShaderResourceTable>(mixed_alias_rt);
        mixed_alias_item.launch.threads_x = W;
        mixed_alias_item.launch.local_x = 64;
        mixed_alias_item.launch.groups_x = 1;
        mixed_alias_item.launch.local_y = mixed_alias_item.launch.local_z = 1;
        mixed_alias_item.launch.groups_y = mixed_alias_item.launch.groups_z = 1;
        mixed_alias_item.code_addr = 0x719dcd;
        CHECK(prosper::frontend::execute_live_compute_items({mixed_alias_item}),
              "mixed compressed/uncompressed storage views execute independently");
        CHECK(std::all_of(mixed_alias_metadata.begin(), mixed_alias_metadata.end(),
                          [](uint8_t code) { return code == 0xff; }),
              "compressed storage alias retains its DCC writeback obligation");
    }

    // Exercise the same production path with a tiled 2D guest surface. The kernel copies row zero;
    // every untouched row must survive upload/readback/retiling exactly. This guards the inverse
    // address mapping used by live tiled StorageImage dispatches, not only the pure tile helper.
    constexpr uint32_t TILED_H = 19;
    constexpr uint32_t tiled2d_mode = static_cast<uint32_t>(TileMode::Sw4KbS);
    std::vector<uint8_t> tiled2d_src_linear(W * TILED_H * 4);
    std::vector<uint8_t> tiled2d_dst_initial(W * TILED_H * 4, 0x5a);
    for (size_t i = 0; i < tiled2d_src_linear.size(); ++i)
        tiled2d_src_linear[i] = static_cast<uint8_t>(i * 41 + 7);
    const size_t tiled2d_bytes = tiled_surface_bytes(W, TILED_H, tiled2d_mode, 0, 4);
    std::vector<uint8_t> tiled2d_src(tiled2d_bytes, 0), tiled2d_dst(tiled2d_bytes, 0);
    tile_surface(tiled2d_src.data(), tiled2d_src_linear.data(), W, TILED_H,
                 tiled2d_mode, 0, 4);
    tile_surface(tiled2d_dst.data(), tiled2d_dst_initial.data(), W, TILED_H,
                 tiled2d_mode, 0, 4);

    ShaderResourceTable tiled2d_rt = irt;
    for (ShaderResource& resource : tiled2d_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 1;
        resource.width = W;
        resource.height = TILED_H;
        resource.depth = 1;
        resource.tile_mode = tiled2d_mode;
        resource.size = static_cast<uint32_t>(tiled2d_bytes);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? tiled2d_src.data() : tiled2d_dst.data());
    }
    const std::vector<uint32_t> tiled2d_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &tiled2d_rt);
    CHECK(!tiled2d_spirv.empty(), "tiled 2D storage-image copy kernel recompiles");
    if (!tiled2d_spirv.empty()) {
        ComputeItem tiled2d_item;
        tiled2d_item.spirv = tiled2d_spirv;
        tiled2d_item.resources = std::make_shared<ShaderResourceTable>(tiled2d_rt);
        tiled2d_item.launch.threads_x = W;
        tiled2d_item.launch.local_x = 64;
        tiled2d_item.launch.groups_x = 1;
        tiled2d_item.launch.local_y = tiled2d_item.launch.local_z = 1;
        tiled2d_item.launch.groups_y = tiled2d_item.launch.groups_z = 1;
        tiled2d_item.code_addr = 0x590592;
        CHECK(prosper::frontend::execute_live_compute_items({tiled2d_item}),
              "production backend executes a tiled 2D storage-image dispatch");

        std::vector<uint8_t> tiled2d_result(W * TILED_H * 4, 0);
        detile_surface(tiled2d_result.data(), tiled2d_dst.data(), W, TILED_H,
                       tiled2d_mode, 0, 4);
        std::vector<uint8_t> tiled2d_expected = tiled2d_dst_initial;
        std::copy_n(tiled2d_src_linear.begin(), W * 4, tiled2d_expected.begin());
        CHECK(tiled2d_result == tiled2d_expected,
              "tiled 2D storage-image writeback matches the linear reference byte-exactly");
    }

    // Some descriptors declare a one-layer 2D array while the MIMG instruction itself uses DIM=2D.
    // Preserve that established byte-identical lowering instead of creating a 2D-array Vulkan view
    // that disagrees with the generated SPIR-V image type.
    std::fill(tiled2d_dst.begin(), tiled2d_dst.end(), 0);
    tile_surface(tiled2d_dst.data(), tiled2d_dst_initial.data(), W, TILED_H,
                 tiled2d_mode, 0, 4);
    ShaderResourceTable single_layer_array_rt = tiled2d_rt;
    for (ShaderResource& resource : single_layer_array_rt.resources)
        if (resource.binding == 4 || resource.binding == 5) resource.img_dim = 5;
    const std::vector<uint32_t> single_layer_array_spirv = recompile_valu(
        image_copy_2d, std::size(image_copy_2d), 1, 0, &single_layer_array_rt);
    CHECK(!single_layer_array_spirv.empty(),
          "DIM=2D kernel recompiles over a byte-identical one-layer array descriptor");
    if (!single_layer_array_spirv.empty()) {
        ComputeItem item;
        item.spirv = single_layer_array_spirv;
        item.resources = std::make_shared<ShaderResourceTable>(single_layer_array_rt);
        item.launch.threads_x = W;
        item.launch.local_x = 64;
        item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x590599;
        CHECK(prosper::frontend::execute_live_compute_items({item}),
              "one-layer array descriptor executes through its DIM=2D Vulkan view");
        std::vector<uint8_t> actual(W * TILED_H * 4, 0);
        detile_surface(actual.data(), tiled2d_dst.data(), W, TILED_H, tiled2d_mode, 0, 4);
        std::vector<uint8_t> expected = tiled2d_dst_initial;
        std::copy_n(tiled2d_src_linear.begin(), W * 4, expected.begin());
        CHECK(actual == expected,
              "one-layer array descriptor retains byte-exact 2D storage writeback");
    }

    // A guest allocation may be cube-shaped while the instruction and generated SPIR-V deliberately
    // view only face zero as a plain 2D texture. This is a view contract, not a request to upload all
    // six faces into a 2D Vulkan image. Plucky's lighting setup uses exactly this form.
    std::vector<uint8_t> cube_faces(tiled2d_bytes * 6u, 0xa7);
    std::copy(tiled2d_src.begin(), tiled2d_src.end(), cube_faces.begin());
    std::fill(tiled2d_dst.begin(), tiled2d_dst.end(), 0);
    tile_surface(tiled2d_dst.data(), tiled2d_dst_initial.data(), W, TILED_H,
                 tiled2d_mode, 0, 4);
    ShaderResourceTable cube_face_rt = tiled2d_rt;
    for (ShaderResource& resource : cube_face_rt.resources) {
        if (resource.binding == 4) {
            resource.cls = ResourceClass::Texture;
            resource.img_dim = 3;
            resource.depth = 6;
            resource.size = static_cast<uint32_t>(cube_faces.size());
            resource.gpu_addr = reinterpret_cast<uint64_t>(cube_faces.data());
            resource.swizzle[0] = 4;
            resource.swizzle[1] = 5;
            resource.swizzle[2] = 6;
            resource.swizzle[3] = 7;
        } else if (resource.binding == 5) {
            resource.gpu_addr = reinterpret_cast<uint64_t>(tiled2d_dst.data());
        }
    }
    const std::vector<uint32_t> cube_face_spirv = recompile_valu(
        image_copy_2d, std::size(image_copy_2d), 1, 0, &cube_face_rt);
    CHECK(!cube_face_spirv.empty(), "DIM=2D kernel recompiles over a cube allocation");
    const auto cube_face_report = validate_spirv_descriptor_interface(
        cube_face_spirv, &cube_face_rt, 0, SpirvShaderStage::Compute);
    const auto cube_face_descriptor = std::find_if(
        cube_face_report.descriptors.begin(), cube_face_report.descriptors.end(),
        [](const SpirvDescriptorBinding& descriptor) { return descriptor.binding == 4; });
    CHECK(cube_face_report.ok() && cube_face_descriptor != cube_face_report.descriptors.end() &&
              cube_face_descriptor->image_dim == 1 && !cube_face_descriptor->image_arrayed,
          "generated shader reflects the cube allocation binding as a non-array 2D image");
    if (!cube_face_spirv.empty()) {
        ComputeItem item;
        item.spirv = cube_face_spirv;
        item.resources = std::make_shared<ShaderResourceTable>(cube_face_rt);
        item.launch.threads_x = W;
        item.launch.local_x = 64;
        item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x59059a;
        CHECK(prosper::frontend::execute_live_compute_items({item}),
              "production backend binds cube face zero through the shader-declared 2D view");
        std::vector<uint8_t> actual(W * TILED_H * 4, 0);
        detile_surface(actual.data(), tiled2d_dst.data(), W, TILED_H, tiled2d_mode, 0, 4);
        std::vector<uint8_t> expected = tiled2d_dst_initial;
        std::copy_n(tiled2d_src_linear.begin(), W * 4, expected.begin());
        CHECK(actual == expected,
              "cube face zero reaches the 2D shader view without reading later faces");
    }

    // Plucky's lighting producer writes a true 3D SW_64KB_S RGBA16 surface. Exercise the full
    // Vulkan storage-image path, including S3 detile/upload and readback/retile, while preserving
    // every voxel outside the row touched by this small copy kernel.
    constexpr uint32_t VOLUME_W = W, VOLUME_H = 32, VOLUME_D = 32;
    constexpr uint32_t volume_mode = static_cast<uint32_t>(TileMode::Sw64KbS);
    constexpr uint32_t volume_bpe = 8;
    const size_t volume_linear_bytes =
        static_cast<size_t>(VOLUME_W) * VOLUME_H * VOLUME_D * volume_bpe;
    std::vector<uint8_t> volume_src_linear(volume_linear_bytes);
    std::vector<uint8_t> volume_dst_initial(volume_linear_bytes);
    for (size_t i = 0; i < volume_linear_bytes / 2; ++i) {
        const uint16_t src_value = static_cast<uint16_t>((i * 313u) & 0xffffu);
        const uint16_t dst_value = static_cast<uint16_t>((i * 197u + 17u) & 0xffffu);
        std::memcpy(volume_src_linear.data() + i * 2, &src_value, 2);
        std::memcpy(volume_dst_initial.data() + i * 2, &dst_value, 2);
    }
    const size_t volume_tiled_bytes = tiled_volume_bytes(
        VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
    std::vector<uint8_t> volume_src(volume_tiled_bytes, 0);
    std::vector<uint8_t> volume_dst(volume_tiled_bytes, 0);
    tile_volume(volume_src.data(), volume_src.size(), volume_src_linear.data(),
                VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
    tile_volume(volume_dst.data(), volume_dst.size(), volume_dst_initial.data(),
                VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
    ShaderResourceTable volume_rt = tiled2d_rt;
    for (ShaderResource& resource : volume_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 2;
        resource.format = DataFormat::Unorm16;
        resource.num_components = 4;
        resource.width = VOLUME_W;
        resource.height = VOLUME_H;
        resource.depth = VOLUME_D;
        resource.tile_mode = volume_mode;
        resource.size = static_cast<uint32_t>(volume_tiled_bytes);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? volume_src.data() : volume_dst.data());
    }
    const std::vector<uint32_t> volume_spirv = recompile_valu(
        image_copy_3d, std::size(image_copy_3d), 1, 0, &volume_rt);
    CHECK(!volume_spirv.empty(), "SW_64KB_S3 storage-volume copy kernel recompiles");
    if (!volume_spirv.empty()) {
        ComputeItem volume_item;
        volume_item.spirv = volume_spirv;
        volume_item.resources = std::make_shared<ShaderResourceTable>(volume_rt);
        volume_item.launch.threads_x = VOLUME_W;
        volume_item.launch.local_x = 64;
        volume_item.launch.groups_x = 1;
        volume_item.launch.local_y = volume_item.launch.local_z = 1;
        volume_item.launch.groups_y = volume_item.launch.groups_z = 1;
        volume_item.code_addr = 0x306a150000;
        CHECK(prosper::frontend::execute_live_compute_items({volume_item}),
              "production backend executes an SW_64KB_S3 storage-volume dispatch");
        std::vector<uint8_t> volume_result(volume_linear_bytes, 0);
        detile_volume(volume_result.data(), volume_dst.data(), volume_dst.size(),
                      VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
        std::vector<uint8_t> volume_expected = volume_dst_initial;
        std::copy_n(volume_src_linear.begin(), VOLUME_W * volume_bpe,
                    volume_expected.begin());
        // #1681: this pair failed on the CI runner's Mesa 25.2.8 lavapipe while passing on Mesa
        // 26.1.4 lavapipe and on RADV, and a bare vector compare cannot say why. Report whether the
        // difference lands in the one row the dispatch writes or in the voxels it must preserve —
        // "wrote the wrong data" and "failed to preserve untouched data" are different defects.
        const size_t volume_written_bytes = VOLUME_W * volume_bpe;
        auto report_volume_diff = [&](const char* tag) {
            if (volume_result == volume_expected) return;
            size_t diff = 0, in_written = 0, first = volume_result.size();
            for (size_t i = 0; i < volume_result.size() && i < volume_expected.size(); ++i) {
                if (volume_result[i] == volume_expected[i]) continue;
                ++diff;
                if (i < volume_written_bytes) ++in_written;
                if (first == volume_result.size()) first = i;
            }
            std::printf("  volume diff [%s]: bytes=%zu/%zu in-written-row=%zu untouched=%zu "
                        "first=%zu got=%02x want=%02x\n",
                        tag, diff, volume_result.size(), in_written, diff - in_written, first,
                        first < volume_result.size() ? volume_result[first] : 0,
                        first < volume_expected.size() ? volume_expected[first] : 0);
        };
        report_volume_diff("S3 writeback");
        CHECK(volume_result == volume_expected,
              "S3 writeback updates one row and preserves all untouched 3D voxels");

        // Renderer ownership is a concrete 2D image identity, not merely an address. Simulate a
        // stale/recycled 2D cache entry at the volume destination and require the layered descriptor
        // to use its valid guest backing without trying to read an impossible 2D snapshot.
        tile_volume(volume_dst.data(), volume_dst.size(), volume_dst_initial.data(),
                    VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
        const uint64_t volume_dst_addr = reinterpret_cast<uint64_t>(volume_dst.data());
        bool layered_reader_called = false;
        set_live_target_query(
            [volume_dst_addr](uint64_t addr) { return addr == volume_dst_addr; });
        set_live_target_reader(
            [&](uint64_t, LiveTargetSnapshot&) {
                layered_reader_called = true;
                return false;
            });
        CHECK(prosper::frontend::execute_live_compute_items({volume_item}),
              "layered resource executes when its base aliases a cached 2D render target");
        std::fill(volume_result.begin(), volume_result.end(), 0);
        detile_volume(volume_result.data(), volume_dst.data(), volume_dst.size(),
                      VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
        report_volume_diff("3D descriptor");
        CHECK(!layered_reader_called && volume_result == volume_expected,
              "3D descriptor rejects address-only 2D ownership and preserves guest-backed voxels");
        set_live_target_reader({});
        set_live_target_query({});
    }

    // A selected mip of a 2D array is not stored as selected-level0, selected-level1, ... . Every
    // array slice owns a complete tail-first mip chain. Exercise a real DIM=2D_ARRAY Vulkan image,
    // seed all three selected subresources through their stride/offset, modify layer two, then prove
    // writeback preserves the other layers and every byte outside the selected level.
    constexpr uint32_t ARRAY_LAYERS = 3;
    constexpr uint32_t ARRAY_BASE_W = W * 2;
    constexpr uint32_t ARRAY_BASE_H = TILED_H * 2;
    constexpr uint32_t ARRAY_MAX_MIP = 4;
    constexpr uint32_t ARRAY_LEVEL = 1;
    const TiledMipLevelLayout array_level = tiled_mip_level_layout(
        ARRAY_BASE_W, ARRAY_BASE_H, 4, tiled2d_mode, ARRAY_MAX_MIP, ARRAY_LEVEL);
    const size_t array_stride = tiled_mip_chain_bytes(
        ARRAY_BASE_W, ARRAY_BASE_H, 4, tiled2d_mode, ARRAY_MAX_MIP);
    const size_t array_selected_bytes = tiled_surface_bytes(W, TILED_H, tiled2d_mode, 0, 4);
    std::vector<uint8_t> array_src(array_stride * ARRAY_LAYERS, 0x31);
    std::vector<uint8_t> array_dst(array_stride * ARRAY_LAYERS, 0xA7);
    std::vector<std::vector<uint8_t>> array_src_linear(
        ARRAY_LAYERS, std::vector<uint8_t>(W * TILED_H * 4));
    std::vector<std::vector<uint8_t>> array_dst_initial(
        ARRAY_LAYERS, std::vector<uint8_t>(W * TILED_H * 4));
    for (uint32_t layer = 0; layer < ARRAY_LAYERS; ++layer) {
        for (size_t i = 0; i < array_src_linear[layer].size(); ++i) {
            array_src_linear[layer][i] = static_cast<uint8_t>(i * 17 + layer * 53 + 5);
            array_dst_initial[layer][i] = static_cast<uint8_t>(i * 29 + layer * 71 + 11);
        }
        tile_surface(array_src.data() + layer * array_stride + array_level.byte_offset,
                     array_src_linear[layer].data(), W, TILED_H, tiled2d_mode, 0, 4);
        tile_surface(array_dst.data() + layer * array_stride + array_level.byte_offset,
                     array_dst_initial[layer].data(), W, TILED_H, tiled2d_mode, 0, 4);
    }
    const std::vector<uint8_t> array_dst_outside_before = array_dst;
    ShaderResourceTable array_rt = tiled2d_rt;
    for (ShaderResource& resource : array_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 5;
        resource.depth = ARRAY_LAYERS;
        resource.size = W * TILED_H * ARRAY_LAYERS * 4;
        resource.layer_stride_bytes = static_cast<uint32_t>(array_stride);
        resource.layer_mip_offset_bytes = static_cast<uint32_t>(array_level.byte_offset);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? array_src.data() : array_dst.data());
    }
    const std::vector<uint32_t> array_spirv = recompile_valu(
        image_copy_2d_array, std::size(image_copy_2d_array), 1, 0, &array_rt);
    CHECK(array_level.supported && !array_level.in_tail && array_stride != 0 &&
              !array_spirv.empty(),
          "selected-mip 2D-array storage copy recompiles with a proven slice layout");
    if (!array_spirv.empty()) {
        ComputeItem array_item;
        array_item.spirv = array_spirv;
        array_item.resources = std::make_shared<ShaderResourceTable>(array_rt);
        array_item.launch.threads_x = W;
        array_item.launch.local_x = 64;
        array_item.launch.groups_x = 1;
        array_item.launch.local_y = array_item.launch.local_z = 1;
        array_item.launch.groups_y = array_item.launch.groups_z = 1;
        array_item.code_addr = 0x590597;
        CHECK(prosper::frontend::execute_live_compute_items({array_item}),
              "production backend executes a selected-mip 2D-array storage dispatch");
        bool selected_levels_exact = true;
        for (uint32_t layer = 0; layer < ARRAY_LAYERS; ++layer) {
            std::vector<uint8_t> actual(W * TILED_H * 4, 0);
            detile_surface(actual.data(),
                           array_dst.data() + layer * array_stride + array_level.byte_offset,
                           W, TILED_H, tiled2d_mode, 0, 4);
            std::vector<uint8_t> expected = array_dst_initial[layer];
            if (layer == 2)
                std::copy_n(array_src_linear[2].begin(), W * 4, expected.begin());
            selected_levels_exact &= actual == expected;
        }
        CHECK(selected_levels_exact,
              "array writeback updates addressed layer two and preserves sibling selected mips");
        bool outside_unchanged = true;
        for (uint32_t layer = 0; layer < ARRAY_LAYERS; ++layer) {
            const size_t selected_begin = layer * array_stride + array_level.byte_offset;
            const size_t selected_end = selected_begin + array_selected_bytes;
            for (size_t i = layer * array_stride; i < (layer + 1) * array_stride; ++i)
                if ((i < selected_begin || i >= selected_end) &&
                    array_dst[i] != array_dst_outside_before[i])
                    outside_unchanged = false;
        }
        CHECK(outside_unchanged,
              "array writeback leaves sibling mips and inter-level padding byte-exact");
    }

    // Linear mip chains use a 256-byte-aligned row pitch independently inside every slice. Use a
    // 65-pixel selected level (260 tight bytes, 512-byte guest pitch) so a bulk tight memcpy would
    // visibly cross both row and layer boundaries. The exact-backing comparison also proves that
    // row padding and sibling levels survive writeback.
    constexpr uint32_t LINEAR_W = W + 1;
    const TiledMipLevelLayout linear_array_level = tiled_mip_level_layout(
        LINEAR_W * 2, ARRAY_BASE_H, 4, 0, ARRAY_MAX_MIP, ARRAY_LEVEL);
    const size_t linear_array_stride = tiled_mip_chain_bytes(
        LINEAR_W * 2, ARRAY_BASE_H, 4, 0, ARRAY_MAX_MIP);
    const size_t linear_array_pitch = linear_sampled_row_pitch(LINEAR_W, 4);
    std::vector<uint8_t> linear_array_src(linear_array_stride * ARRAY_LAYERS, 0x43);
    std::vector<uint8_t> linear_array_dst(linear_array_stride * ARRAY_LAYERS, 0xB9);
    std::vector<uint8_t> linear_array_expected = linear_array_dst;
    for (uint32_t layer = 0; layer < ARRAY_LAYERS; ++layer) {
        for (uint32_t y = 0; y < TILED_H; ++y) {
            uint8_t* src_row = linear_array_src.data() + layer * linear_array_stride +
                linear_array_level.byte_offset + y * linear_array_pitch;
            uint8_t* dst_row = linear_array_dst.data() + layer * linear_array_stride +
                linear_array_level.byte_offset + y * linear_array_pitch;
            for (uint32_t x = 0; x < LINEAR_W * 4; ++x) {
                src_row[x] = static_cast<uint8_t>(x * 17 + y * 31 + layer * 59 + 3);
                dst_row[x] = static_cast<uint8_t>(x * 23 + y * 37 + layer * 67 + 9);
            }
        }
    }
    linear_array_expected = linear_array_dst;
    const size_t linear_array_layer_two = 2u * linear_array_stride;
    std::memcpy(linear_array_expected.data() + linear_array_layer_two +
                    linear_array_level.byte_offset,
                linear_array_src.data() + linear_array_layer_two +
                    linear_array_level.byte_offset,
                W * 4);
    ShaderResourceTable linear_array_rt = tiled2d_rt;
    for (ShaderResource& resource : linear_array_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 5;
        resource.width = LINEAR_W;
        resource.height = TILED_H;
        resource.depth = ARRAY_LAYERS;
        resource.tile_mode = 0;
        resource.size = LINEAR_W * TILED_H * ARRAY_LAYERS * 4;
        resource.layer_stride_bytes = static_cast<uint32_t>(linear_array_stride);
        resource.layer_mip_offset_bytes = static_cast<uint32_t>(linear_array_level.byte_offset);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? linear_array_src.data() : linear_array_dst.data());
    }
    const std::vector<uint32_t> linear_array_spirv = recompile_valu(
        image_copy_2d_array, std::size(image_copy_2d_array), 1, 0, &linear_array_rt);
    CHECK(linear_array_level.supported && linear_array_stride != 0 &&
              linear_array_pitch == 512 && !linear_array_spirv.empty(),
          "linear selected-mip array exposes its aligned per-row and per-slice layout");
    if (!linear_array_spirv.empty()) {
        ComputeItem linear_array_item;
        linear_array_item.spirv = linear_array_spirv;
        linear_array_item.resources = std::make_shared<ShaderResourceTable>(linear_array_rt);
        linear_array_item.launch.threads_x = W;
        linear_array_item.launch.local_x = 64;
        linear_array_item.launch.groups_x = 1;
        linear_array_item.launch.local_y = linear_array_item.launch.local_z = 1;
        linear_array_item.launch.groups_y = linear_array_item.launch.groups_z = 1;
        linear_array_item.code_addr = 0x590598;
        CHECK(prosper::frontend::execute_live_compute_items({linear_array_item}),
              "production backend executes a selected-mip linear 2D-array dispatch");
        CHECK(linear_array_dst == linear_array_expected,
              "linear array writeback preserves aligned rows, sibling slices, mips, and padding");
    }

    // Exercise the wider element mappings that become default-on with tiled storage. Float16 uses
    // finite values so half->float->half is exact; Float32 follows the backend's raw channel contract.
    auto wide_tiled_roundtrip = [&](DataFormat format, uint32_t bytes_per_texel,
                                    uint64_t code_addr) {
        const size_t linear_bytes = static_cast<size_t>(W) * TILED_H * bytes_per_texel;
        std::vector<uint8_t> src_linear(linear_bytes, 0);
        std::vector<uint8_t> dst_initial(linear_bytes, 0);
        if (format == DataFormat::Float16) {
            for (size_t i = 0; i < linear_bytes / 2; ++i) {
                const uint16_t src_half = float_to_half(
                    static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f);
                const uint16_t dst_half = float_to_half(
                    static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f);
                std::memcpy(src_linear.data() + i * 2, &src_half, sizeof(src_half));
                std::memcpy(dst_initial.data() + i * 2, &dst_half, sizeof(dst_half));
            }
        } else {
            for (size_t i = 0; i < linear_bytes / 4; ++i) {
                const uint32_t src_word = static_cast<uint32_t>(i * 2654435761u + 0x1020304u);
                const uint32_t dst_word = static_cast<uint32_t>(i * 2246822519u + 0x5060708u);
                std::memcpy(src_linear.data() + i * 4, &src_word, sizeof(src_word));
                std::memcpy(dst_initial.data() + i * 4, &dst_word, sizeof(dst_word));
            }
        }
        const size_t tiled_bytes_wide = tiled_surface_bytes(
            W, TILED_H, tiled2d_mode, 0, bytes_per_texel);
        std::vector<uint8_t> src_tiled(tiled_bytes_wide, 0);
        std::vector<uint8_t> dst_tiled(tiled_bytes_wide, 0);
        tile_surface(src_tiled.data(), src_linear.data(), W, TILED_H,
                     tiled2d_mode, 0, bytes_per_texel);
        tile_surface(dst_tiled.data(), dst_initial.data(), W, TILED_H,
                     tiled2d_mode, 0, bytes_per_texel);
        ShaderResourceTable wide_rt = tiled2d_rt;
        for (ShaderResource& resource : wide_rt.resources) {
            if (resource.binding != 4 && resource.binding != 5) continue;
            resource.format = format;
            resource.num_components = 4;
            resource.size = static_cast<uint32_t>(tiled_bytes_wide);
            resource.gpu_addr = reinterpret_cast<uint64_t>(
                resource.binding == 4 ? src_tiled.data() : dst_tiled.data());
        }
        const std::vector<uint32_t> wide_spirv = recompile_valu(
            image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
            &wide_rt);
        if (wide_spirv.empty()) return false;
        ComputeItem wide_item;
        wide_item.spirv = wide_spirv;
        wide_item.resources = std::make_shared<ShaderResourceTable>(wide_rt);
        wide_item.launch.threads_x = W;
        wide_item.launch.local_x = 64;
        wide_item.launch.groups_x = 1;
        wide_item.launch.local_y = wide_item.launch.local_z = 1;
        wide_item.launch.groups_y = wide_item.launch.groups_z = 1;
        wide_item.code_addr = code_addr;
        if (!prosper::frontend::execute_live_compute_items({wide_item})) return false;
        std::vector<uint8_t> result(linear_bytes, 0);
        detile_surface(result.data(), dst_tiled.data(), W, TILED_H,
                       tiled2d_mode, 0, bytes_per_texel);
        std::vector<uint8_t> expected = dst_initial;
        std::copy_n(src_linear.begin(), static_cast<size_t>(W) * bytes_per_texel,
                    expected.begin());
        return result == expected;
    };
    CHECK(wide_tiled_roundtrip(DataFormat::Float16, 8, 0x590595),
          "tiled Float16x4 storage writeback is byte-exact");
    CHECK(wide_tiled_roundtrip(DataFormat::Float32, 16, 0x590596),
          "tiled Float32x4 storage writeback is byte-exact");

    ShaderResourceTable unsupported_tiled_rt = tiled2d_rt;
    for (ShaderResource& resource : unsupported_tiled_rt.resources)
        if (resource.binding == 4 || resource.binding == 5) resource.tile_mode = 6;
    const std::vector<uint32_t> unsupported_tiled_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &unsupported_tiled_rt);
    CHECK(!unsupported_tiled_spirv.empty(), "unsupported tiled storage kernel still recompiles");
    if (!unsupported_tiled_spirv.empty()) {
        ComputeItem unsupported_tiled_item;
        unsupported_tiled_item.spirv = unsupported_tiled_spirv;
        unsupported_tiled_item.resources =
            std::make_shared<ShaderResourceTable>(unsupported_tiled_rt);
        unsupported_tiled_item.launch.threads_x = W;
        unsupported_tiled_item.launch.local_x = 64;
        unsupported_tiled_item.launch.groups_x = 1;
        unsupported_tiled_item.launch.local_y = unsupported_tiled_item.launch.local_z = 1;
        unsupported_tiled_item.launch.groups_y = unsupported_tiled_item.launch.groups_z = 1;
        unsupported_tiled_item.code_addr = 0x590594;
        std::vector<ComputeAuthorityBoundary> failed_compute_boundaries;
        set_compute_authority_boundary_observer(
            [&](const ComputeAuthorityBoundary& boundary) {
                failed_compute_boundaries.push_back(boundary);
            });
        const bool unsupported_executed =
            prosper::frontend::execute_live_compute_items({unsupported_tiled_item});
        set_compute_authority_boundary_observer({});
        CHECK(!unsupported_executed && failed_compute_boundaries.size() == 1 &&
                  failed_compute_boundaries[0].kind ==
                      ComputeAuthorityBoundaryKind::Compute &&
                  !failed_compute_boundaries[0].range_known,
              "failed live compute emits one fail-closed boundary before returning false");
    }

    // A renderer-owned color target is newer than its guest backing. Recompile the same copy kernel
    // with a sampled source, publish deliberately different live pixels, and prove the production
    // backend imports the immutable renderer snapshot instead of either skipping or reading stale RAM.
    std::vector<uint8_t> stale_rtt(W * 4, 0);
    auto live_rtt = std::make_shared<std::vector<uint8_t>>(W * 4);
    std::vector<uint8_t> live_dst(W * 4, 0xEE);
    for (uint32_t i = 0; i < W * 4; ++i) (*live_rtt)[i] = static_cast<uint8_t>(i * 29 + 11);
    ShaderResourceTable live_rt = irt;
    for (ShaderResource& resource : live_rt.resources) {
        if (resource.binding == 4) {
            resource.cls = ResourceClass::Texture;
            resource.img_dim = 1;
            resource.gpu_addr = reinterpret_cast<uint64_t>(stale_rtt.data());
            resource.host_data = nullptr;
            resource.host_data_size = 0;
            resource.swizzle[0] = 4;
            resource.swizzle[1] = 5;
            resource.swizzle[2] = 6;
            resource.swizzle[3] = 7;
        } else if (resource.binding == 5) {
            resource.img_dim = 1;
            resource.gpu_addr = reinterpret_cast<uint64_t>(live_dst.data());
            resource.host_data = nullptr;
            resource.host_data_size = 0;
        }
    }
    const uint64_t live_addr = reinterpret_cast<uint64_t>(stale_rtt.data());
    set_live_target_query([live_addr](uint64_t addr) { return addr == live_addr; });
    set_live_target_reader(
        [live_addr, live_rtt](uint64_t addr, LiveTargetSnapshot& snapshot) {
            if (addr != live_addr) return false;
            snapshot.width = W;
            snapshot.height = 1;
            snapshot.format = LiveTargetPixelFormat::Rgba8Unorm;
            snapshot.pixels = live_rtt;
            return true;
        });
    std::vector<uint32_t> live_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &live_rt);
    CHECK(!live_spirv.empty(), "sampled-image copy kernel recompiles for renderer RTT import");
    if (!live_spirv.empty()) {
        ComputeItem live_item;
        live_item.spirv = live_spirv;
        live_item.resources = std::make_shared<ShaderResourceTable>(live_rt);
        live_item.launch.threads_x = W;
        live_item.launch.local_x = 64;
        live_item.launch.groups_x = 1;
        live_item.launch.local_y = live_item.launch.local_z = 1;
        live_item.launch.groups_y = live_item.launch.groups_z = 1;
        live_item.code_addr = 0x590591;
        CHECK(prosper::frontend::execute_live_compute_items({live_item}),
              "live backend executes a dispatch sampling a renderer-owned RTT");
        CHECK(live_dst == *live_rtt,
              "sampled renderer RTT pixels reach storage-image writeback byte-exactly");
        CHECK(live_dst != stale_rtt, "renderer RTT import does not use stale guest backing");
    }

    // GTA V Fidelity retains single-channel R16F post-process targets in the renderer and later
    // samples them from compute. The CPU fallback must bind the snapshot at its native texel width:
    // treating R16F as the historical RGBA8 fallback doubled the memcpy byte count, read past the
    // immutable snapshot, and crashed Windows in ucrtbase!memmove before gameplay. RG16F has the
    // same byte count as RGBA8 but still needs its native numeric interpretation, so cover both.
    auto run_live_narrow_float16 = [&](uint32_t components,
                                       LiveTargetPixelFormat pixel_format) {
        const uint32_t source_bytes = W * components * sizeof(uint16_t);
        std::vector<uint8_t> stale(source_bytes, 0);
        auto live = std::make_shared<std::vector<uint8_t>>(source_bytes);
        std::vector<uint8_t> destination(W * 4, 0xee);
        for (uint32_t texel = 0; texel < W; ++texel) {
            for (uint32_t channel = 0; channel < components; ++channel) {
                const float value = ((texel >> channel) & 1u) ? 1.0f : 0.0f;
                const uint16_t half = float_to_half(value);
                std::memcpy(live->data() +
                                (texel * components + channel) * sizeof(half),
                            &half, sizeof(half));
            }
        }
        ShaderResourceTable table = live_rt;
        for (ShaderResource& resource : table.resources) {
            if (resource.binding == 4) {
                resource.cls = ResourceClass::Texture;
                resource.format = DataFormat::Float16;
                resource.num_components = components;
                resource.width = W;
                resource.height = resource.depth = 1;
                resource.gpu_addr = reinterpret_cast<uint64_t>(stale.data());
                resource.size = source_bytes;
            } else if (resource.binding == 5) {
                resource.cls = ResourceClass::StorageImage;
                resource.format = DataFormat::Unorm8;
                resource.num_components = 4;
                resource.width = W;
                resource.height = resource.depth = 1;
                resource.gpu_addr = reinterpret_cast<uint64_t>(destination.data());
                resource.size = static_cast<uint32_t>(destination.size());
            }
        }
        const uint64_t address = reinterpret_cast<uint64_t>(stale.data());
        set_live_target_query([address](uint64_t candidate) { return candidate == address; });
        set_live_target_reader(
            [address, live, pixel_format, W](uint64_t candidate, LiveTargetSnapshot& snapshot) {
                if (candidate != address) return false;
                snapshot.width = W;
                snapshot.height = 1;
                snapshot.format = pixel_format;
                snapshot.pixels = live;
                return true;
            });
        const std::vector<uint32_t> spirv = recompile_valu(
            image_copy_2d, std::size(image_copy_2d), 1, 0, &table);
        CHECK(!spirv.empty(), components == 1
              ? "R16F renderer RTT sampled-image copy kernel recompiles"
              : "RG16F renderer RTT sampled-image copy kernel recompiles");
        bool executed = false;
        if (!spirv.empty()) {
            ComputeItem item;
            item.spirv = spirv;
            item.resources = std::make_shared<ShaderResourceTable>(table);
            item.launch.threads_x = W;
            item.launch.local_x = 64;
            item.launch.groups_x = 1;
            item.launch.local_y = item.launch.local_z = 1;
            item.launch.groups_y = item.launch.groups_z = 1;
            item.code_addr = components == 1 ? 0x5905b0 : 0x5905b1;
            executed = prosper::frontend::execute_live_compute_items({item});
        }
        CHECK(executed, components == 1
              ? "live backend samples an R16F renderer RTT without widening its upload"
              : "live backend samples an RG16F renderer RTT at native width");
        bool exact = executed;
        for (uint32_t texel = 0; texel < W && exact; ++texel) {
            exact &= destination[texel * 4 + 0] == ((texel & 1u) ? 255u : 0u);
            exact &= destination[texel * 4 + 1] ==
                (components == 2 && (texel & 2u) ? 255u : 0u);
            exact &= destination[texel * 4 + 2] == 0u;
            exact &= destination[texel * 4 + 3] == 255u;
        }
        CHECK(exact, components == 1
              ? "R16F renderer snapshot preserves R and Vulkan missing-channel defaults"
              : "RG16F renderer snapshot preserves RG and Vulkan missing-channel defaults");
    };
    run_live_narrow_float16(1, LiveTargetPixelFormat::R16Float);
    run_live_narrow_float16(2, LiveTargetPixelFormat::Rg16Float);

    // A renderer target can become a writable storage image before its pixels have been materialized
    // in guest RAM. Seed the dispatch from the immutable renderer snapshot, modify row zero, and
    // prove writeback preserves every untouched row while publishing a cache-invalidation write.
    std::vector<uint8_t> writable_rtt_guest(tiled2d_bytes, 0);
    auto writable_rtt = std::make_shared<std::vector<uint8_t>>(W * TILED_H * 4);
    for (size_t i = 0; i < writable_rtt->size(); ++i)
        (*writable_rtt)[i] = static_cast<uint8_t>(i * 17 + 3);
    ShaderResourceTable writable_rt = tiled2d_rt;
    for (ShaderResource& resource : writable_rt.resources) {
        if (resource.binding == 5) {
            resource.gpu_addr = reinterpret_cast<uint64_t>(writable_rtt_guest.data());
            resource.size = static_cast<uint32_t>(writable_rtt_guest.size());
        }
    }
    const uint64_t writable_addr = reinterpret_cast<uint64_t>(writable_rtt_guest.data());
    set_live_target_query([writable_addr](uint64_t addr) { return addr == writable_addr; });
    set_live_target_reader(
        [writable_addr, writable_rtt](uint64_t addr, LiveTargetSnapshot& snapshot) {
            if (addr != writable_addr) return false;
            snapshot.width = W;
            snapshot.height = TILED_H;
            snapshot.format = LiveTargetPixelFormat::Rgba8Unorm;
            snapshot.pixels = writable_rtt;
            return true;
        });
    bool writable_rtt_published = false;
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char*) {
        writable_rtt_published |= addr == writable_addr && size == writable_rtt_guest.size();
    });
    const std::vector<uint32_t> writable_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &writable_rt);
    CHECK(!writable_spirv.empty(), "writable renderer RTT copy kernel recompiles");
    if (!writable_spirv.empty()) {
        ComputeItem writable_item;
        writable_item.spirv = writable_spirv;
        writable_item.resources = std::make_shared<ShaderResourceTable>(writable_rt);
        writable_item.launch.threads_x = W;
        writable_item.launch.local_x = 64;
        writable_item.launch.groups_x = 1;
        writable_item.launch.local_y = writable_item.launch.local_z = 1;
        writable_item.launch.groups_y = writable_item.launch.groups_z = 1;
        writable_item.code_addr = 0x590593;
        CHECK(prosper::frontend::execute_live_compute_items({writable_item}),
              "live backend executes a dispatch writing a renderer-owned RTT");
        std::vector<uint8_t> writable_result(W * TILED_H * 4, 0);
        detile_surface(writable_result.data(), writable_rtt_guest.data(), W, TILED_H,
                       tiled2d_mode, 0, 4);
        std::vector<uint8_t> writable_expected = *writable_rtt;
        std::copy_n(tiled2d_src_linear.begin(), W * 4, writable_expected.begin());
        CHECK(writable_result == writable_expected,
              "writable renderer RTT seeds from live pixels and preserves untouched rows");
        CHECK(writable_rtt_published,
              "writable renderer RTT publishes guest writeback for cache invalidation");
    }
    set_guest_gpu_write_observer({});
    set_live_target_reader({});
    set_live_target_query({});

    // Exact write-only storage aliases must fold before their independent RTT seed preparation.
    // The two stores cover different rows through different bindings: losing either store or
    // splitting the Vulkan image cannot satisfy the byte oracle. Only the canonical owner proves
    // coverage, so an alias must not repeatedly request a snapshot for its never-published proof.
    auto check_write_only_rtt_alias = [&](uint32_t height, uint64_t code_addr) {
        const bool full = height == 2;
        const bool allow_warm_seed_skip =
            std::getenv("PROSPER_NO_SKIP_SEED") == nullptr &&
            std::getenv("PROSPER_VERIFY_SEED_SKIP") == nullptr &&
            prosper::frontend::seed_reprove_interval_from_env(
                std::getenv("PROSPER_SEED_REPROVE"), 256u) != 1;
        static const uint32_t alias_rows[] = {
            0x7E080300u,              // v4 = x from the local ID
            0x7E0A0280u,              // v5 = 0: owner writes row zero
            0x7E0002F2u,              // v0 = 1.0f (red)
            0x7E020280u,              // v1 = 0
            0x7E040280u,              // v2 = 0
            0x7E0602F2u,              // v3 = 1.0f (alpha)
            0xF0200F08u, 0x00020004u, // IMAGE_STORE through s[8:15], binding 5
            0x7E0A0281u,              // v5 = 1: alias writes row one
            0x7E000280u,              // v0 = 0
            0x7E0202F2u,              // v1 = 1.0f (green)
            0xF0200F08u, 0x00040004u, // IMAGE_STORE through s[16:23], binding 6
            0xBF810000u,
        };
        std::vector<uint8_t> destination(static_cast<size_t>(W) * height * 4, 0xC7);
        auto snapshot_bytes = std::make_shared<std::vector<uint8_t>>(destination.size());
        for (size_t byte = 0; byte < snapshot_bytes->size(); ++byte)
            (*snapshot_bytes)[byte] = static_cast<uint8_t>(byte * 13 + 29);
        const uint64_t address = reinterpret_cast<uint64_t>(destination.data());
        ShaderResourceTable table;
        for (uint32_t binding = 5; binding <= 6; ++binding) {
            ShaderResource resource{};
            resource.cls = ResourceClass::StorageImage;
            resource.img_dim = 1;
            resource.binding = binding;
            resource.sgpr_base = binding == 5 ? 8 : 16;
            resource.format = DataFormat::Unorm8;
            resource.num_components = 4;
            resource.width = W;
            resource.height = height;
            resource.depth = 1;
            resource.gpu_addr = address;
            resource.size = static_cast<uint32_t>(destination.size());
            table.resources.push_back(resource);
        }
        ComputeShaderConfig config;
        config.user_sgprs.resize(24);
        config.local_x = W;
        config.local_y = config.local_z = 1;
        config.tidig_comp_cnt = 0;
        config.native_storage_format_support =
            native_storage_format_support_bit(DataFormat::Unorm8, 4);
        const std::vector<uint32_t> spirv =
            recompile_compute(alias_rows, std::size(alias_rows), &table, config);
        const DescriptorValidationReport report = validate_spirv_descriptor_interface(
            spirv, &table, 0, SpirvShaderStage::Compute, false);
        const bool write_only_aliases = !spirv.empty() && report.ok() &&
            report.descriptors.size() == 2 &&
            std::all_of(report.descriptors.begin(), report.descriptors.end(),
                [](const SpirvDescriptorBinding& descriptor) {
                    return descriptor.kind == SpirvDescriptorKind::StorageImage &&
                           descriptor.writable && !descriptor.readable &&
                           descriptor.storage_float;
                });
        CHECK(write_only_aliases,
              "RTT alias fixture reflects two native write-only storage bindings");
        if (!write_only_aliases) return;

        ComputeItem item;
        item.spirv = spirv;
        item.resources = std::make_shared<ShaderResourceTable>(table);
        item.launch.threads_x = item.launch.local_x = W;
        item.launch.groups_x = 1;
        item.launch.threads_y = item.launch.threads_z = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = code_addr;
        uint32_t queries = 0, reads = 0, notifications = 0;
        bool fail_reader = false;
        set_live_target_query([&](uint64_t candidate) {
            if (candidate != address) return false;
            ++queries;
            return true;
        });
        set_live_target_reader([&](uint64_t candidate, LiveTargetSnapshot& snapshot) {
            if (candidate != address) return false;
            ++reads;
            if (fail_reader) return false;
            snapshot.width = W;
            snapshot.height = height;
            snapshot.format = LiveTargetPixelFormat::Rgba8Unorm;
            snapshot.pixels = snapshot_bytes;
            return true;
        });
        set_guest_gpu_write_observer([&](uint64_t written_addr, uint64_t bytes, const char*) {
            if (written_addr == address && bytes == destination.size()) ++notifications;
        });
        for (uint32_t run = 0; run < 2; ++run) {
            const bool unused_reader = full && run != 0 && allow_warm_seed_skip;
            queries = reads = notifications = 0;
            // A fresh guest mirror must not satisfy the result oracle without actual writeback.
            std::fill(destination.begin(), destination.end(), 0xC7);
            if (run) {
                for (uint8_t& byte : *snapshot_bytes) byte ^= 0x6D;
            }
            fail_reader = unused_reader;
            std::vector<uint8_t> expected = *snapshot_bytes;
            for (uint32_t x = 0; x < W; ++x) {
                const size_t first = static_cast<size_t>(x) * 4;
                const size_t second = (static_cast<size_t>(W) + x) * 4;
                expected[first] = 255; expected[first + 1] = expected[first + 2] = 0;
                expected[first + 3] = 255;
                expected[second] = expected[second + 2] = 0;
                expected[second + 1] = expected[second + 3] = 255;
            }
            const bool executed = prosper::frontend::execute_live_compute_items({item});
            CHECK(executed, unused_reader
                  ? "warm full RTT aliases execute even when an unused snapshot reader would fail"
                  : "RTT aliases execute with the required canonical snapshot seed");
            CHECK(destination == expected, full
                  ? "both write-only aliases contribute their exact distinct output rows"
                  : "partial write-only aliases preserve the untouched row from the current RTT seed");
            CHECK(reads == (unused_reader ? 0u : 1u), unused_reader
                  ? "warm full RTT aliases request no discarded CPU snapshot"
                  : "RTT alias proving or partial run requests exactly one owner snapshot");
            CHECK(queries == 2,
                  "both RTT storage bindings retain their ownership query and pending-write drain");
            CHECK(notifications == 1,
                  "exact RTT storage aliases publish one canonical guest-write notification");
        }
        if (!full) {
            const std::vector<uint8_t> before_failure = destination;
            reads = notifications = 0;
            fail_reader = true;
            CHECK(!prosper::frontend::execute_live_compute_items({item}),
                  "partial RTT alias dispatch still rejects a missing authoritative seed");
            CHECK(reads == 1 && notifications == 0 && destination == before_failure,
                  "failed partial RTT alias seed leaves guest bytes and write notifications untouched");
        }
        set_guest_gpu_write_observer({});
        set_live_target_reader({});
        set_live_target_query({});
    };
    check_write_only_rtt_alias(2, 0x590EA10u);
    check_write_only_rtt_alias(3, 0x590EA11u);

    // #3391: an exact storage view is one image even when its bindings have different access.
    // Each invocation owns one column, so reads of the untouched row (partial) or reads before
    // that invocation's own stores (full) have no cross-invocation race. The SSBO independently
    // observes the input: restoring poisoned pixels at writeback cannot repair a wrong shader read.
    auto check_mixed_storage_alias = [&](bool reader_first, bool full, bool join_after_proof) {
        const uint32_t writer_binding = reader_first ? 6u : 5u;
        const uint32_t reader_binding = reader_first ? 5u : 6u;
        constexpr uint32_t stored = 0x13579BDFu;
        std::vector<uint32_t> shader = {
            0x7E080300u, // v4 = local x, preserved across image loads
        };
        auto read_row = [&](uint32_t row, uint32_t buffer_offset) {
            shader.insert(shader.end(), {
                row ? 0x7E0A0281u : 0x7E0A0280u, // v5 = row
                0xF0000108u, 0x00040804u,       // IMAGE_LOAD x -> v8, s[16:23]
                0xBF8C3F70u,                    // s_waitcnt vmcnt(0)
                0xE0702000u | buffer_offset, 0x80000804u, // store v8 at v4, s[0:3]
            });
        };
        auto write_row = [&](uint32_t row) {
            shader.insert(shader.end(), {
                row ? 0x7E0A0281u : 0x7E0A0280u, // v5 = row
                0x7E0002FFu, stored,             // v0 = exact integer literal
                0xF0200108u, 0x00020004u,        // IMAGE_STORE x, s[8:15]
            });
        };
        if (full) {
            read_row(0, 0);
            write_row(0);
            read_row(1, W * sizeof(uint32_t));
            write_row(1);
        } else {
            read_row(1, 0);
            write_row(0);
        }
        shader.push_back(0xBF810000u);
        std::vector<uint32_t> image_words(W * 2);
        std::vector<uint32_t> separate_reader(W * 2);
        std::vector<uint32_t> observed(W * (full ? 2u : 1u), 0xCCCCCCCCu);
        ShaderResourceTable table;
        ShaderResource buffer{};
        buffer.cls = ResourceClass::ConstantBuffer;
        buffer.binding = 2;
        buffer.sgpr_base = 0;
        buffer.format = DataFormat::Uint32;
        buffer.num_components = 1;
        buffer.stride = sizeof(uint32_t);
        buffer.gpu_addr = reinterpret_cast<uint64_t>(observed.data());
        buffer.size = static_cast<uint32_t>(observed.size() * sizeof(uint32_t));
        table.resources.push_back(buffer);
        for (uint32_t binding = 5; binding <= 6; ++binding) {
            ShaderResource image{};
            image.cls = ResourceClass::StorageImage;
            image.binding = binding;
            image.sgpr_base = binding == writer_binding ? 8 : 16;
            image.img_dim = 1;
            image.format = DataFormat::Uint32;
            image.num_components = 1;
            image.width = W;
            image.height = 2;
            image.depth = 1;
            image.gpu_addr = reinterpret_cast<uint64_t>(image_words.data());
            image.size = static_cast<uint32_t>(image_words.size() * sizeof(uint32_t));
            table.resources.push_back(image);
        }
        ComputeShaderConfig config;
        config.user_sgprs.resize(24);
        config.local_x = W;
        config.local_y = config.local_z = 1;
        config.tidig_comp_cnt = 0;
        config.native_storage_format_support =
            native_storage_format_support_bit(DataFormat::Uint32, 1);
        const std::vector<uint32_t> spirv =
            recompile_compute(shader.data(), shader.size(), &table, config);
        const DescriptorValidationReport report = validate_spirv_descriptor_interface(
            spirv, &table, 0, SpirvShaderStage::Compute, false);
        const auto* writer = find_spirv_descriptor_binding(report, 0, writer_binding);
        const auto* reader = find_spirv_descriptor_binding(report, 0, reader_binding);
        const auto* output = find_spirv_descriptor_binding(report, 0, buffer.binding);
        const auto exact_r32 = [](const SpirvDescriptorBinding* descriptor) {
            return descriptor && descriptor->kind == SpirvDescriptorKind::StorageImage &&
                   descriptor->image_numeric_class == SpirvImageNumericClass::Uint &&
                   descriptor->storage_image_format == kSpirvImageFormatR32ui &&
                   descriptor->image_dim == 1 && !descriptor->image_arrayed &&
                   !descriptor->image_multisampled;
        };
        const bool access_ok = !spirv.empty() && report.ok() && report.descriptors.size() == 3 &&
            exact_r32(writer) && writer->writable && !writer->readable &&
            exact_r32(reader) && reader->readable && !reader->writable &&
            output && output->kind == SpirvDescriptorKind::StorageBuffer && output->writable;
        CHECK(access_ok, "mixed storage fixture reflects distinct write-only/read-only R32_UINT bindings and SSBO");
        if (!access_ok) return;
        const ComputeImageViewShape shape{true, 1, 1};
        CHECK(shader_resource_same_view(table.resources[1], table.resources[2], shape, shape, true),
              "mixed storage fixture begins with exact storage-view identity");
        ComputeItem item;
        item.spirv = spirv;
        item.launch.threads_x = item.launch.local_x = W;
        item.launch.threads_y = item.launch.threads_z = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x33910000u + (reader_first ? 0x100u : 0u) +
                         (full ? 0x10u : 0u) + (join_after_proof ? 1u : 0u);
        for (uint32_t run = 0; run < 4; ++run) {
            const bool aliased = !join_after_proof || run >= 2;
            for (uint32_t texel = 0; texel < W * 2; ++texel) {
                image_words[texel] = 0x24680000u + run * 0x1000u + texel * 17u;
                separate_reader[texel] = 0xABCD0000u + run * 0x1000u + texel * 19u;
            }
            std::fill(observed.begin(), observed.end(), 0xCCCCCCCCu);
            for (ShaderResource& resource : table.resources) {
                if (resource.binding == reader_binding)
                    resource.gpu_addr = reinterpret_cast<uint64_t>(
                        aliased ? image_words.data() : separate_reader.data());
            }
            CHECK(shader_resource_same_view(table.resources[1], table.resources[2], shape, shape, true) == aliased,
                  "mixed storage dispatch pins exact alias identity or deliberate separate-view control");
            const std::vector<uint32_t>& input = aliased ? image_words : separate_reader;
            const std::vector<uint32_t> expected_observed(
                input.begin() + (full ? 0u : W), input.end());
            std::vector<uint32_t> expected_image = image_words;
            std::fill_n(expected_image.begin(), full ? W * 2 : W, stored);
            item.resources = std::make_shared<ShaderResourceTable>(table);
            const bool executed = prosper::frontend::execute_live_compute_items({item});
            if (!executed || image_words != expected_image || observed != expected_observed)
                std::printf("  mixed storage alias reader_first=%u full=%u join_after_proof=%u run=%u aliased=%u\n",
                            reader_first, full, join_after_proof, run, aliased);
            CHECK(executed, "mixed storage alias dispatch executes in both binding orders");
            CHECK(image_words == expected_image,
                  "mixed storage alias writes exact output and preserves every untouched image texel");
            CHECK(observed == expected_observed,
                  "mixed storage alias reader observes current exact input, including warm and newly joined views");
        }
    };
    for (bool reader_first : {false, true}) {
        check_mixed_storage_alias(reader_first, false, false);
        check_mixed_storage_alias(reader_first, true, false);
        check_mixed_storage_alias(reader_first, true, true);
    }

    // A Full proof established jointly by two write-only aliases is not a proof for either
    // binding alone. Keep code and extents fixed while the group splits, then joins again.
    // Conversely, an isolated owner's None proof cannot suppress a newly joined writer.
    auto check_write_only_group_membership = [&](bool none_owner) {
        std::vector<uint32_t> split_writer_code = {
            0x7E080300u,                         // v4 = x
            0x7E0A0280u,                         // row zero through binding 5
            0x7E0002FFu, 0x13579BDFu,
            0xF0200108u, 0x00020004u,
            0x7E0A0281u,                         // row one through binding 6
            0x7E0002FFu, 0x2468ACE0u,
            0xF0200108u, 0x00040004u,
            0xBF810000u,
        };
        if (none_owner) {
            // Keep the store reflected, but put the owner's row outside the image. Binding 6
            // still writes row one, so separate -> joined changes owner coverage None -> Partial.
            split_writer_code[1] = 0x7E0A02FFu;
            split_writer_code.insert(split_writer_code.begin() + 2, 9999u);
        }
        std::vector<uint32_t> first(W * 2), second(W * 2);
        ShaderResourceTable table;
        for (uint32_t binding = 5; binding <= 6; ++binding) {
            ShaderResource image{};
            image.cls = ResourceClass::StorageImage;
            image.binding = binding;
            image.sgpr_base = binding == 5 ? 8 : 16;
            image.img_dim = 1;
            image.format = DataFormat::Uint32;
            image.num_components = 1;
            image.width = W;
            image.height = 2;
            image.depth = 1;
            image.gpu_addr = reinterpret_cast<uint64_t>(first.data());
            image.size = static_cast<uint32_t>(first.size() * sizeof(uint32_t));
            table.resources.push_back(image);
        }
        ComputeShaderConfig config;
        config.user_sgprs.resize(24);
        config.local_x = W;
        config.local_y = config.local_z = 1;
        config.tidig_comp_cnt = 0;
        config.native_storage_format_support =
            native_storage_format_support_bit(DataFormat::Uint32, 1);
        const std::vector<uint32_t> spirv = recompile_compute(
            split_writer_code.data(), split_writer_code.size(), &table, config);
        const DescriptorValidationReport report = validate_spirv_descriptor_interface(
            spirv, &table, 0, SpirvShaderStage::Compute, false);
        const bool shape_ok = !spirv.empty() && report.ok() && report.descriptors.size() == 2 &&
            std::all_of(report.descriptors.begin(), report.descriptors.end(),
                [](const SpirvDescriptorBinding& descriptor) {
                    return descriptor.kind == SpirvDescriptorKind::StorageImage &&
                           descriptor.writable && !descriptor.readable &&
                           descriptor.image_numeric_class == SpirvImageNumericClass::Uint &&
                           descriptor.storage_image_format == kSpirvImageFormatR32ui &&
                           descriptor.image_dim == 1 && !descriptor.image_arrayed &&
                           !descriptor.image_multisampled;
                });
        CHECK(shape_ok, "alias split/rejoin fixture reflects two exact write-only R32_UINT images");
        if (shape_ok) {
            ComputeItem item;
            item.spirv = spirv;
            item.code_addr = none_owner ? 0x3391A11Bu : 0x3391A11Au;
            item.launch.threads_x = item.launch.local_x = W;
            item.launch.threads_y = item.launch.threads_z = 1;
            item.launch.local_y = item.launch.local_z = 1;
            item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
            const ComputeImageViewShape shape{true, 1, 1};
            for (uint32_t run = 0; run < 6; ++run) {
                const bool aliased = none_owner ? run >= 2 && run < 4 : run < 2 || run >= 4;
                for (uint32_t texel = 0; texel < W * 2; ++texel) {
                    first[texel] = 0xAAAA0000u + run * 0x1000u + texel * 23u;
                    second[texel] = 0xBBBB0000u + run * 0x1000u + texel * 29u;
                }
                table.resources[1].gpu_addr = reinterpret_cast<uint64_t>(
                    aliased ? first.data() : second.data());
                CHECK(shader_resource_same_view(table.resources[0], table.resources[1], shape, shape, true) == aliased,
                      "write-only group membership follows the explicit alias/split/rejoin sequence");
                std::vector<uint32_t> expected_first = first, expected_second = second;
                if (!none_owner) std::fill_n(expected_first.begin(), W, 0x13579BDFu);
                std::fill_n((aliased ? expected_first : expected_second).begin() + W, W, 0x2468ACE0u);
                item.resources = std::make_shared<ShaderResourceTable>(table);
                const bool executed = prosper::frontend::execute_live_compute_items({item});
                if (!executed || first != expected_first || second != expected_second)
                    std::printf("  write-only alias split/rejoin none_owner=%u run=%u aliased=%u\n",
                                none_owner, run, aliased);
                CHECK(executed, "write-only storage group executes through alias/split/rejoin transitions");
                CHECK(first == expected_first && second == expected_second,
                      none_owner
                          ? "write-only group None proof never suppresses a newly joined writer"
                          : "write-only group Full proof never discards fresh untouched rows after a split");
            }
        }
    };
    check_write_only_group_membership(false);
    check_write_only_group_membership(true);

    // The exact packed fallback must obey the same authority rule. Keep stale zeroes in guest RAM,
    // publish distinct R11G11B10 renderer pixels, overwrite only row zero, and require every other
    // row to survive from the snapshot rather than the stale allocation.
    std::vector<uint32_t> packed_rtt_source(W * TILED_H);
    std::vector<uint32_t> packed_rtt_guest(W * TILED_H, 0);
    auto packed_rtt = std::make_shared<std::vector<uint8_t>>(W * TILED_H * sizeof(uint32_t));
    std::vector<uint32_t> packed_rtt_expected(W * TILED_H);
    for (size_t t = 0; t < packed_rtt_source.size(); ++t) {
        packed_rtt_source[t] = static_cast<uint32_t>(float_to_f11((t % 19u) * 0.25f)) |
            (static_cast<uint32_t>(float_to_f11((t % 23u) * 0.375f)) << 11) |
            (static_cast<uint32_t>(float_to_f10((t % 29u) * 0.5f)) << 22);
        const uint32_t live_word = static_cast<uint32_t>(float_to_f11((t % 31u) * 0.125f)) |
            (static_cast<uint32_t>(float_to_f11((t % 37u) * 0.1875f)) << 11) |
            (static_cast<uint32_t>(float_to_f10((t % 41u) * 0.3125f)) << 22);
        std::memcpy(packed_rtt->data() + t * sizeof(live_word), &live_word, sizeof(live_word));
        packed_rtt_expected[t] = live_word;
    }
    std::copy_n(packed_rtt_source.begin(), W, packed_rtt_expected.begin());
    ShaderResourceTable packed_writable_rt = tiled2d_rt;
    for (ShaderResource& resource : packed_writable_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.format = DataFormat::Float10_11_11;
        resource.num_components = 3;
        resource.tile_mode = 0;
        resource.size = W * TILED_H * sizeof(uint32_t);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? packed_rtt_source.data() : packed_rtt_guest.data());
    }
    const uint64_t packed_writable_addr =
        reinterpret_cast<uint64_t>(packed_rtt_guest.data());
    set_live_target_query(
        [packed_writable_addr](uint64_t addr) { return addr == packed_writable_addr; });
    set_live_target_reader(
        [packed_writable_addr, packed_rtt](uint64_t addr, LiveTargetSnapshot& snapshot) {
            if (addr != packed_writable_addr) return false;
            snapshot.width = W;
            snapshot.height = TILED_H;
            snapshot.format = LiveTargetPixelFormat::R11G11B10Float;
            snapshot.pixels = packed_rtt;
            return true;
        });
    const std::vector<uint32_t> packed_writable_spirv = recompile_valu(
        image_copy_2d, std::size(image_copy_2d), 1, 0, &packed_writable_rt);
    CHECK(!packed_writable_spirv.empty(),
          "partial R11G11B10 renderer RTT copy kernel recompiles");
    if (!packed_writable_spirv.empty()) {
        ComputeItem item;
        item.spirv = packed_writable_spirv;
        item.resources = std::make_shared<ShaderResourceTable>(packed_writable_rt);
        item.launch.threads_x = W;
        item.launch.local_x = 64;
        item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x590b12;
        CHECK(prosper::frontend::execute_live_compute_items({item}),
              "live backend partially writes an authoritative packed renderer RTT");
        CHECK(packed_rtt_guest == packed_rtt_expected,
              "packed renderer RTT seeds from live pixels and preserves every untouched row");
        CHECK(std::any_of(packed_rtt_guest.begin() + W, packed_rtt_guest.end(),
                          [](uint32_t word) { return word != 0; }),
              "packed renderer RTT never falls back to its stale zeroed guest backing");
    }
    set_live_target_reader({});
    set_live_target_query({});

    // UE4's exposure chain samples tiny tiled Float32 surfaces. Preserve those values natively:
    // normalizing through RGBA8 would clamp negative and HDR channels before the compute shader sees
    // them. Copy a tiled Float32x4 source through the production sampled-image path into the existing
    // raw-channel Float32 storage path and require exact bits at the guest destination.
    std::vector<float> float_src(W * 4), float_dst(W * 4, 0.0f);
    for (uint32_t i = 0; i < W * 4; ++i)
        float_src[i] = (static_cast<int32_t>(i % 17) - 8) * 0.375f;
    const size_t float_tiled_bytes = tiled_surface_bytes(W, 1, dcc_tile, 0, 16);
    std::vector<uint8_t> float_tiled(float_tiled_bytes, 0);
    tile_surface(float_tiled.data(), reinterpret_cast<const uint8_t*>(float_src.data()),
                 W, 1, dcc_tile, 0, 16);
    ShaderResourceTable float_rt = irt;
    for (ShaderResource& resource : float_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 1;
        resource.format = DataFormat::Float32;
        resource.num_components = 4;
        resource.width = W;
        resource.height = resource.depth = 1;
        if (resource.binding == 4) {
            resource.cls = ResourceClass::Texture;
            resource.tile_mode = dcc_tile;
            resource.gpu_addr = reinterpret_cast<uint64_t>(float_tiled.data());
            resource.size = static_cast<uint32_t>(float_tiled.size());
            resource.swizzle[0] = 4;
            resource.swizzle[1] = 5;
            resource.swizzle[2] = 6;
            resource.swizzle[3] = 7;
        } else {
            resource.tile_mode = 0;
            resource.gpu_addr = reinterpret_cast<uint64_t>(float_dst.data());
            resource.size = static_cast<uint32_t>(float_dst.size() * sizeof(float));
        }
    }
    std::vector<uint32_t> float_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0, &float_rt);
    CHECK(!float_spirv.empty(), "sampled-image copy kernel recompiles for tiled Float32 input");
    if (!float_spirv.empty()) {
        ComputeItem float_item;
        float_item.spirv = float_spirv;
        float_item.resources = std::make_shared<ShaderResourceTable>(float_rt);
        float_item.launch.threads_x = W;
        float_item.launch.local_x = 64;
        float_item.launch.groups_x = 1;
        float_item.launch.local_y = float_item.launch.local_z = 1;
        float_item.launch.groups_y = float_item.launch.groups_z = 1;
        float_item.code_addr = 0x590f32;
        CHECK(prosper::frontend::execute_live_compute_items({float_item}),
              "live backend executes a dispatch sampling a tiled Float32 image");
        CHECK(float_dst == float_src,
              "tiled Float32 sampled values preserve negative and HDR channels byte-exactly");
    }

    // Sonic's full-screen compute resolve reads two unsigned tiled views that were previously
    // rejected by the live uploader: R32_UINT and RGBA16_UINT.  Keep these as native integer Vulkan
    // images so image_load returns exact component bits; routing them through RGBA8 would truncate
    // the former and reinterpret the latter as normalized color.
    auto sampled_uint_roundtrip = [&](DataFormat format, uint32_t components,
                                      uint32_t bytes_per_texel, uint64_t code_addr) {
        std::vector<uint8_t> linear_src(W * bytes_per_texel);
        std::vector<uint8_t> linear_dst(W * bytes_per_texel, 0xA5);
        for (size_t i = 0; i < linear_src.size(); ++i)
            linear_src[i] = static_cast<uint8_t>(i * 73 + 19);
        const size_t tiled_size = tiled_surface_bytes(W, 1, dcc_tile, 0, bytes_per_texel);
        std::vector<uint8_t> tiled_src(tiled_size, 0);
        tile_surface(tiled_src.data(), linear_src.data(), W, 1, dcc_tile, 0, bytes_per_texel);
        ShaderResourceTable uint_rt = irt;
        for (ShaderResource& resource : uint_rt.resources) {
            if (resource.binding != 4 && resource.binding != 5) continue;
            resource.img_dim = 1;
            resource.format = format;
            resource.num_components = components;
            resource.width = W;
            resource.height = resource.depth = 1;
            if (resource.binding == 4) {
                resource.cls = ResourceClass::Texture;
                resource.tile_mode = dcc_tile;
                resource.gpu_addr = reinterpret_cast<uint64_t>(tiled_src.data());
                resource.size = static_cast<uint32_t>(tiled_src.size());
                resource.swizzle[0] = 4;
                resource.swizzle[1] = 5;
                resource.swizzle[2] = 6;
                resource.swizzle[3] = 7;
            } else {
                resource.tile_mode = 0;
                resource.gpu_addr = reinterpret_cast<uint64_t>(linear_dst.data());
                resource.size = static_cast<uint32_t>(linear_dst.size());
            }
        }
        const std::vector<uint32_t> spv = recompile_valu(
            image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0, &uint_rt);
        if (spv.empty()) return false;
        ComputeItem item;
        item.spirv = spv;
        item.resources = std::make_shared<ShaderResourceTable>(uint_rt);
        item.launch.threads_x = W;
        item.launch.local_x = 64;
        item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = code_addr;
        return prosper::frontend::execute_live_compute_items({item}) && linear_dst == linear_src;
    };
    CHECK(sampled_uint_roundtrip(DataFormat::Uint32, 1, 4, 0x590320),
          "tiled R32_UINT sampled values reach storage writeback byte-exactly");
    CHECK(sampled_uint_roundtrip(DataFormat::Uint16, 4, 8, 0x590164),
          "tiled RGBA16_UINT sampled values reach storage writeback byte-exactly");
    CHECK(sampled_uint_roundtrip(DataFormat::Unorm2_10_10_10, 4, 4, 0x590210),
          "tiled R10G10B10A2_UNORM sampled words reach storage writeback bit-exactly");

    // GPU captures preserve descriptor addresses but materialize resource bytes in owned host
    // arrays. Warm replay must retain those sampled images just like the live guest-backed path,
    // while validating by exact bytes because host_data mutations never enter the guest journal or
    // page write-watch system. Prove both the unchanged hit and a direct unreported mutation.
    {
        constexpr uint32_t host_texel_bytes = 4;
        std::vector<uint8_t> linear_src(W * host_texel_bytes);
        std::vector<uint8_t> linear_dst(W * host_texel_bytes, 0xA5);
        for (size_t i = 0; i < linear_src.size(); ++i)
            linear_src[i] = static_cast<uint8_t>(i * 41 + 13);
        const size_t tiled_size = tiled_surface_bytes(
            W, 1, dcc_tile, 0, host_texel_bytes);
        std::vector<uint8_t> capture_owned_src(tiled_size, 0);
        tile_surface(capture_owned_src.data(), linear_src.data(), W, 1,
                     dcc_tile, 0, host_texel_bytes);

        ShaderResourceTable host_rt = irt;
        for (ShaderResource& resource : host_rt.resources) {
            if (resource.binding != 4 && resource.binding != 5) continue;
            resource.img_dim = 1;
            resource.format = DataFormat::Uint8;
            resource.num_components = 4;
            resource.width = W;
            resource.height = resource.depth = 1;
            if (resource.binding == 4) {
                resource.cls = ResourceClass::Texture;
                resource.tile_mode = dcc_tile;
                resource.gpu_addr = 0x71c0000000ull;
                resource.size = static_cast<uint32_t>(capture_owned_src.size());
                resource.host_data = capture_owned_src.data();
                resource.host_data_size = capture_owned_src.size();
                resource.swizzle[0] = 4;
                resource.swizzle[1] = 5;
                resource.swizzle[2] = 6;
                resource.swizzle[3] = 7;
            } else {
                resource.tile_mode = 0;
                resource.gpu_addr = reinterpret_cast<uint64_t>(linear_dst.data());
                resource.size = static_cast<uint32_t>(linear_dst.size());
            }
        }
        const std::vector<uint32_t> host_spirv = recompile_valu(
            image_copy_2d, std::size(image_copy_2d), 1, 0, &host_rt);
        CHECK(!host_spirv.empty(),
              "capture-owned sampled-image copy kernel recompiles");
        if (!host_spirv.empty()) {
            ComputeItem host_item;
            host_item.spirv = host_spirv;
            host_item.resources = std::make_shared<ShaderResourceTable>(host_rt);
            host_item.launch.threads_x = W;
            host_item.launch.local_x = 64;
            host_item.launch.groups_x = 1;
            host_item.launch.local_y = host_item.launch.local_z = 1;
            host_item.launch.groups_y = host_item.launch.groups_z = 1;
            host_item.code_addr = 0x590ca9;
            CHECK(prosper::frontend::execute_live_compute_items({host_item}) &&
                      linear_dst == linear_src,
                  "capture-owned sampled image establishes an exact retained source");

            const uint64_t skips_before =
                prosper::frontend::live_compute_sampled_image_upload_skips();
            CHECK(prosper::frontend::execute_live_compute_items({host_item}) &&
                      prosper::frontend::live_compute_sampled_image_upload_skips() >
                          skips_before,
                  "unchanged capture-owned sampled image skips its warm-replay upload");

            linear_src[0] ^= 0xff;
            tile_surface(capture_owned_src.data(), linear_src.data(), W, 1,
                         dcc_tile, 0, host_texel_bytes);
            const uint64_t skips_before_mutation =
                prosper::frontend::live_compute_sampled_image_upload_skips();
            CHECK(prosper::frontend::execute_live_compute_items({host_item}) &&
                      linear_dst == linear_src &&
                      prosper::frontend::live_compute_sampled_image_upload_skips() ==
                          skips_before_mutation,
                  "direct capture-owned mutation refreshes the retained sampled image exactly");
        }
    }

    // --- #1122 seed-skip coverage proof. A write-only storage image whose dispatch fully covers the
    // extent can skip the (expensive) seed -- BUT ONLY after proving the write actually stores every
    // texel. covers_extent is necessary, not sufficient: a shader that stores a SUBSET of a covering
    // grid leaves the rest undefined, and skipping the seed there packs pool garbage to the guest.
    // The backend proves coverage per (shader,binding) on first sight (poison the seed, require zero
    // survivors) and only fast-skips proven-full shaders; a partial write is detected and always
    // seeds, restoring untouched texels. These two cases pin both halves of that contract.

    // (a) FULL coverage: a 1D write-only fill (no image load) over threads_x == width. The seed is
    // unobservable, so the poison-proving run and every seed-skipped run must agree byte-for-byte and
    // differ from the pre-run guest content (the kernel really wrote every texel).
    {
        static const uint32_t fill_1d[] = {
            0x7E080300u,             // v4 = v0 (x coord from the shell input)
            0x7E0A0301u,             // v5 = v1 (y local ID; zero for the one-row launch)
            0x7E040280u,             // v2 = 0 (stored B)
            0x7E060280u,             // v3 = 0 (stored A)
            0xF0200F08u, 0x00020004u,// IMAGE_STORE v0..v3 at (v4,v5) -- NO preceding IMAGE_LOAD
            0xBF810000u,             // s_endpgm
        };
        // A typed RGBA16F storage image exercises the same native-storage retention path as the
        // game's 4K post-process output. Height one keeps the coverage proof compact.
        const size_t fill_guest_bytes = W * 8;
        const size_t fill_mapping_bytes = 4096;
        std::vector<uint8_t> fill_guest_storage;
        uint8_t* fill_guest = nullptr;
#if defined(__linux__)
        void* fill_mapping = mmap(nullptr, fill_mapping_bytes, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(fill_mapping != MAP_FAILED, "map typed storage-image regression range");
        if (fill_mapping == MAP_FAILED) return fails ? fails : 1;
        fill_guest = static_cast<uint8_t*>(fill_mapping);
        prosper::host::guest_write_watch_set_fault_onstack(true);
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(fill_guest), fill_mapping_bytes, 0x7c0000,
            0x3 /* SCE CPU_READ|CPU_WRITE */);
#else
        fill_guest_storage.resize(fill_guest_bytes);
        fill_guest = fill_guest_storage.data();
#endif
        std::fill_n(fill_guest, fill_guest_bytes, 0xC3);   // distinctive pre-run content
        const std::vector<uint8_t> fill_original(fill_guest, fill_guest + fill_guest_bytes);
        auto fill_equals = [&](const std::vector<uint8_t>& expected) {
            return expected.size() == fill_guest_bytes &&
                   std::equal(fill_guest, fill_guest + fill_guest_bytes, expected.begin());
        };
        ShaderResourceTable fill_rt;
        ShaderResource fdst{};
        fdst.cls = ResourceClass::StorageImage; fdst.img_dim = 1; fdst.binding = 5; fdst.sgpr_base = 8;
        fdst.format = DataFormat::Float16; fdst.num_components = 4; fdst.width = W; fdst.height = 1;
        fdst.depth = 1; fdst.gpu_addr = (uint64_t)(uintptr_t)fill_guest;
        fdst.size = static_cast<uint32_t>(fill_guest_bytes);
        fill_rt.resources.push_back(fdst);
        ComputeShaderConfig fill_config;
        fill_config.user_sgprs.resize(16);
        fill_config.local_x = W;
        fill_config.local_y = fill_config.local_z = 1;
        fill_config.tidig_comp_cnt = 1;
        fill_config.native_storage_format_support =
            native_storage_format_support_bit(DataFormat::Float16, 4);
        std::vector<uint32_t> fill_spirv = recompile_compute(
            fill_1d, sizeof(fill_1d) / sizeof(fill_1d[0]), &fill_rt, fill_config);
        CHECK(!fill_spirv.empty(), "write-only 1D fill kernel recompiles");
        if (!fill_spirv.empty()) {
            ComputeItem it; it.spirv = fill_spirv;
            it.resources = std::make_shared<ShaderResourceTable>(fill_rt);
            it.launch.threads_x = W; it.launch.local_x = 64; it.launch.groups_x = 1;
            it.launch.threads_y = it.launch.threads_z = 1;
            it.launch.local_y = it.launch.local_z = 1;
            it.launch.groups_y = it.launch.groups_z = 1;
            it.code_addr = 0x1122f11du;   // fresh code -> first run proves, second run seed-skips
            CHECK(prosper::frontend::execute_live_compute_items({it}),
                  "seed-skip proving run (poison-seeded) executes a full-coverage 1D fill");
            const std::vector<uint8_t> after_prove(fill_guest, fill_guest + fill_guest_bytes);
            CHECK(!fill_equals(fill_original),
                  "full-coverage fill overwrites the guest content (kernel ran)");
            std::fill_n(fill_guest, fill_guest_bytes, 0x00);  // scrub between runs
            CHECK(prosper::frontend::execute_live_compute_items({it}),
                  "seed-skip fast run (seed skipped) executes the proven full-coverage fill");
            CHECK(fill_equals(after_prove),
                  "seed-skipped run is byte-identical to the poison-proven run (seed is unobserved)");

#if defined(__linux__)
            auto repeated_write_watch = prosper::host::GuestWriteWatch::create(
                reinterpret_cast<uint64_t>(fill_guest), fill_guest_bytes);
            const uint64_t repeated_snapshots_before =
                prosper::frontend::live_compute_storage_result_snapshot_bytes();
#endif
            uint32_t repeated_write_notifications = 0;
            set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char*) {
                if (addr == fdst.gpu_addr && size == fdst.size)
                    ++repeated_write_notifications;
            });
            CHECK(prosper::frontend::execute_live_compute_items({it}),
                  "retained full-coverage image repeats an identical dispatch");
            set_guest_gpu_write_observer({});
            CHECK(fill_equals(after_prove),
                  "identical retained output leaves the exact guest result intact");
            CHECK(repeated_write_notifications == 1,
                  "identical retained output invalidates renderer aliases without rewriting bytes");
#if defined(__linux__)
            CHECK(static_cast<bool>(repeated_write_watch) && repeated_write_watch.query() ==
                      prosper::host::GuestWriteWatchQuery::Unchanged,
                  "identical retained output leaves guest-byte dirty tracking clean");
            CHECK(prosper::frontend::live_compute_storage_result_snapshot_bytes() ==
                      repeated_snapshots_before,
                  "identical retained output does not recopy its guest-byte baseline");
            repeated_write_watch.reset();
#endif

            // The live draw immediately following a compute producer is inside the same ordered
            // guest submit. Its mutation journal is an exact authority even before a cross-submit
            // page watch has accumulated enough stable validations to be promoted.
            std::vector<uint8_t> journal_guest(W * 8, 0x51);
            ShaderResourceTable journal_rt = fill_rt;
            ShaderResource& journal_dst = journal_rt.resources.back();
            journal_dst.gpu_addr = reinterpret_cast<uint64_t>(journal_guest.data());
            ComputeItem journal_item = it;
            journal_item.resources = std::make_shared<ShaderResourceTable>(journal_rt);
            journal_item.dispatch_index = 31;
            journal_item.command_order = 10;
            DrawItem journal_consumer;
            journal_consumer.draw_index = 47;
            journal_consumer.command_order = 20;
            bool journal_imported = false;
            const OrderedSubmitResult journal_result = execute_ordered_items(
                {{SubmitOperationKind::Dispatch, journal_item.dispatch_index,
                  journal_item.command_order},
                 {SubmitOperationKind::Draw, journal_consumer.draw_index,
                  journal_consumer.command_order}},
                {journal_consumer}, {journal_item},
                [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                    ShaderResource sampled = journal_dst;
                    sampled.cls = ResourceClass::Texture;
                    prosper::frontend::LiveComputeImageImport compute_import;
                    journal_imported = prosper::frontend::import_live_compute_storage_image(
                        sampled, journal_dst.size, compute_import) && compute_import.valid();
                    return RenderedFrame{};
                },
                [&](const std::vector<ComputeItem>& items) {
                    return prosper::frontend::execute_live_compute_items(items);
                },
                1, 1);
            CHECK(journal_result.compute_executed && journal_result.render_spans == 1 &&
                      journal_imported,
                  "same-submit journal authorizes the first retained compute-image consumer");

            // GTA V's transition depth-like surface is written through R32_UINT storage and then
            // sampled as R32_SFLOAT. The guest allocation is one exact 32-bit word per texel; only
            // the consumer view changes its numeric interpretation. Prove the integer producer can
            // be leased by the float descriptor without a guest readback/detile/upload round trip.
            std::vector<uint32_t> uint_alias_guest(W, 0x3f000000u);
            ShaderResourceTable uint_alias_rt;
            ShaderResource uint_alias_dst = fdst;
            uint_alias_dst.format = DataFormat::Uint32;
            uint_alias_dst.num_components = 1;
            uint_alias_dst.gpu_addr = reinterpret_cast<uint64_t>(uint_alias_guest.data());
            uint_alias_dst.size = static_cast<uint32_t>(uint_alias_guest.size() * sizeof(uint32_t));
            uint_alias_rt.resources.push_back(uint_alias_dst);
            ComputeShaderConfig uint_alias_config = fill_config;
            uint_alias_config.native_storage_format_support =
                native_storage_format_support_bit(DataFormat::Uint32, 1);
            const std::vector<uint32_t> uint_alias_spirv = recompile_compute(
                fill_1d, std::size(fill_1d), &uint_alias_rt, uint_alias_config);
            const DescriptorValidationReport uint_alias_report =
                validate_spirv_descriptor_interface(
                    uint_alias_spirv, &uint_alias_rt, 0, SpirvShaderStage::Compute, false);
            const SpirvDescriptorBinding* uint_alias_binding =
                find_spirv_descriptor_binding(
                    uint_alias_report, 0, uint_alias_dst.binding);
            CHECK(!uint_alias_spirv.empty() && uint_alias_report.ok() && uint_alias_binding &&
                      uint_alias_binding->image_numeric_class == SpirvImageNumericClass::Uint &&
                      uint_alias_binding->storage_image_format == kSpirvImageFormatR32ui,
                  "R32_UINT alias producer retains an exact typed storage contract");
            if (!uint_alias_spirv.empty() && uint_alias_report.ok() && uint_alias_binding) {
                ComputeItem uint_alias_item = it;
                uint_alias_item.spirv = uint_alias_spirv;
                uint_alias_item.resources =
                    std::make_shared<ShaderResourceTable>(uint_alias_rt);
                uint_alias_item.code_addr = 0x1122f13du;
                uint_alias_item.dispatch_index = 35;
                uint_alias_item.command_order = 10;
                CHECK(prosper::frontend::execute_live_compute_items({uint_alias_item}),
                      "R32_UINT alias producer proves complete write coverage");
                CHECK(prosper::frontend::execute_live_compute_items({uint_alias_item}),
                      "R32_UINT alias producer repeats after its full-coverage proof");
                ShaderResource cross_submit_float_alias = uint_alias_dst;
                cross_submit_float_alias.cls = ResourceClass::Texture;
                cross_submit_float_alias.format = DataFormat::Float32;
                prosper::frontend::LiveComputeImageImport cross_submit_import;
                const bool cross_submit_imported =
                    prosper::frontend::import_live_compute_storage_image(
                        cross_submit_float_alias, uint_alias_dst.size,
                        cross_submit_import) && cross_submit_import.valid();
#if defined(_WIN32)
                CHECK(cross_submit_imported,
                      "Windows exact guest mirror authorizes a cross-submit R32_UINT-to-R32_SFLOAT import");
#else
                CHECK(!cross_submit_imported,
                      "Linux does not replace journal/write-watch authority with byte equality");
#endif
                cross_submit_import = {};
                const uint32_t uint_alias_word = uint_alias_guest[1];
                uint_alias_guest[1] ^= 1u;
                prosper::frontend::LiveComputeImageImport rejected_cross_submit_import;
                CHECK(!prosper::frontend::import_live_compute_storage_image(
                          cross_submit_float_alias, uint_alias_dst.size,
                          rejected_cross_submit_import),
                      "an unnotified CPU write rejects the exact cross-submit compute image");
                uint_alias_guest[1] = uint_alias_word;
                bool uint_float_imported = false;
                DrawItem uint_alias_consumer;
                uint_alias_consumer.draw_index = 49;
                uint_alias_consumer.command_order = 20;
                const OrderedSubmitResult uint_alias_result = execute_ordered_items(
                    {{SubmitOperationKind::Dispatch, uint_alias_item.dispatch_index,
                      uint_alias_item.command_order},
                     {SubmitOperationKind::Draw, uint_alias_consumer.draw_index,
                      uint_alias_consumer.command_order}},
                    {uint_alias_consumer}, {uint_alias_item},
                    [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                        ShaderResource sampled_alias = uint_alias_dst;
                        sampled_alias.cls = ResourceClass::Texture;
                        sampled_alias.format = DataFormat::Float32;
                        prosper::frontend::LiveComputeImageImport compute_import;
                        uint_float_imported =
                            prosper::frontend::import_live_compute_storage_image(
                                sampled_alias, uint_alias_dst.size, compute_import) &&
                            compute_import.valid() && compute_import.native_format == 100u;
                        return RenderedFrame{};
                    },
                    [&](const std::vector<ComputeItem>& items) {
                        return prosper::frontend::execute_live_compute_items(items);
                    },
                    1, 1);
                CHECK(uint_alias_result.compute_executed &&
                          uint_alias_result.render_spans == 1 && uint_float_imported,
                      "R32_UINT storage result imports directly through an R32_SFLOAT graphics view");

                // The next GTA V pass consumes the same numeric-view alias from compute, not
                // graphics. Keep the sampled destination separate and prove that the Float32
                // descriptor is seeded by a device-local copy of the retained Uint32 producer.
                // Exact output bits are the oracle: R32_UINT -> R32_SFLOAT must not numerically
                // convert the producer words on the way to the sampled image.
                std::vector<uint32_t> uint_alias_copy_guest(W, 0xdeadbeefu);
                ShaderResourceTable uint_alias_consumer_rt;
                ShaderResource uint_alias_sampled = uint_alias_dst;
                uint_alias_sampled.cls = ResourceClass::Texture;
                uint_alias_sampled.format = DataFormat::Float32;
                uint_alias_sampled.binding = 4;
                uint_alias_sampled.sgpr_base = 0;
                uint_alias_sampled.swizzle[0] = 4;
                uint_alias_sampled.swizzle[1] = 5;
                uint_alias_sampled.swizzle[2] = 6;
                uint_alias_sampled.swizzle[3] = 7;
                uint_alias_consumer_rt.resources.push_back(uint_alias_sampled);
                ShaderResource uint_alias_copy_dst = uint_alias_sampled;
                uint_alias_copy_dst.cls = ResourceClass::StorageImage;
                uint_alias_copy_dst.binding = 5;
                uint_alias_copy_dst.sgpr_base = 8;
                uint_alias_copy_dst.gpu_addr =
                    reinterpret_cast<uint64_t>(uint_alias_copy_guest.data());
                uint_alias_consumer_rt.resources.push_back(uint_alias_copy_dst);
                ComputeShaderConfig uint_alias_consumer_config = fill_config;
                uint_alias_consumer_config.native_storage_format_support =
                    native_storage_format_support_bit(DataFormat::Float32, 1);
                const std::vector<uint32_t> uint_alias_consumer_spirv = recompile_compute(
                    image_copy_2d, std::size(image_copy_2d),
                    &uint_alias_consumer_rt, uint_alias_consumer_config);
                CHECK(!uint_alias_consumer_spirv.empty(),
                      "R32_SFLOAT compute alias consumer recompiles");
                if (!uint_alias_consumer_spirv.empty()) {
                    ComputeItem uint_alias_consumer = uint_alias_item;
                    uint_alias_consumer.spirv = uint_alias_consumer_spirv;
                    uint_alias_consumer.resources =
                        std::make_shared<ShaderResourceTable>(uint_alias_consumer_rt);
                    uint_alias_consumer.code_addr = 0x1122f13eu;
                    uint_alias_consumer.dispatch_index = 36;
                    uint_alias_consumer.command_order = 20;
                    const uint64_t cross_submit_seeds_before =
                        prosper::frontend::live_compute_storage_transfer_seeds();
                    CHECK(prosper::frontend::execute_live_compute_items(
                              {uint_alias_consumer}) &&
                              uint_alias_copy_guest == uint_alias_guest,
                          "cross-submit R32_SFLOAT compute consumer preserves exact producer bits");
#if defined(_WIN32)
                    CHECK(!native_2d_compute_transfer_available ||
                              prosper::frontend::live_compute_storage_transfer_seeds() >
                                  cross_submit_seeds_before,
                          "Windows exact guest mirror seeds the cross-submit compute consumer on-GPU");
#else
                    CHECK(prosper::frontend::live_compute_storage_transfer_seeds() ==
                              cross_submit_seeds_before,
                          "Linux cross-submit consumer keeps the authority-unknown image on guest fallback");
#endif
                    const uint64_t uint_alias_seeds_before =
                        prosper::frontend::live_compute_storage_transfer_seeds();
                    const OrderedSubmitResult uint_alias_compute_result =
                        execute_ordered_items(
                            {{SubmitOperationKind::Dispatch,
                              uint_alias_item.dispatch_index,
                              uint_alias_item.command_order},
                             {SubmitOperationKind::Dispatch,
                              uint_alias_consumer.dispatch_index,
                              uint_alias_consumer.command_order}},
                            {}, {uint_alias_item, uint_alias_consumer},
                            [](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                                return RenderedFrame{};
                            },
                            [&](const std::vector<ComputeItem>& items) {
                                return prosper::frontend::execute_live_compute_items(items);
                            },
                            1, 1);
                    const uint64_t uint_alias_seeds_after =
                        prosper::frontend::live_compute_storage_transfer_seeds();
                    CHECK(uint_alias_compute_result.compute_executed &&
                              uint_alias_copy_guest == uint_alias_guest,
                          "R32_UINT producer to R32_SFLOAT compute consumer preserves exact bits");
                    CHECK(!native_2d_compute_transfer_available ||
                              uint_alias_seeds_after > uint_alias_seeds_before,
                          "R32_UINT producer seeds R32_SFLOAT compute consumer on-GPU");
                    CHECK(native_2d_compute_transfer_available ||
                              uint_alias_seeds_after == uint_alias_seeds_before,
                          "disabled native transfer keeps the numeric-view alias on the guest fallback");
                }
            }

            // Syberia's save-warning pass dispatches eleven shrinking rectangles over one native
            // Float32x1 atlas in a single ordered guest submit. Its producer uses a real one-layer
            // DIM=2D_ARRAY storage image while the next dispatch samples an ordinary DIM=2D view of
            // the same guest allocation. Keep the sampled and writable images distinct
            // (read-old/write-new), but seed the sampled image with a device-local copy of the
            // retained arrayed storage result instead of detiling and uploading the complete guest
            // mirror. This fixture uses the submit journal, rather than a cross-submit page watch,
            // as authority. Pin format equality separately so a future sampled-format policy change
            // cannot silently turn the production check into the guest fallback.
            CHECK(prosper::frontend::compute_native_2d_transfer_format_compatible(
                      DataFormat::Float32, 1) &&
                      prosper::frontend::compute_native_2d_transfer_format_compatible(
                          DataFormat::Float10_11_11, 3) &&
                      !prosper::frontend::compute_native_2d_transfer_format_compatible(
                          DataFormat::Float16, 1),
                  "native 2D transfer admits exact formats and packed R11 bit-copy compatibility");
            {
                prosper::gpu::ShaderResource single_level{};
                single_level.cls = prosper::gpu::ResourceClass::Texture;
                single_level.width = 256;
                single_level.height = 256;
                single_level.img_dim = 1;
                single_level.declared_mip_levels = 1;
                CHECK(prosper::frontend::compute_binding_mip_chain_materializable(single_level, false) &&
                      prosper::frontend::compute_binding_mip_chain_materializable(single_level, true),
                      "single-level compute image binding is viable regardless of render target ownership");

                constexpr uint32_t kWidth = 256, kHeight = 256, kBytesPerBlock = 4;
                constexpr uint32_t kTile = static_cast<uint32_t>(prosper::gpu::TileMode::Sw64KbRX);
                constexpr uint32_t kMaxMip = 8;
                const auto level0 = prosper::gpu::tiled_mip_level_layout(
                    kWidth, kHeight, kBytesPerBlock, kTile, kMaxMip, 0);

                prosper::gpu::ShaderResource chain{};
                chain.cls = prosper::gpu::ResourceClass::Texture;
                chain.format = prosper::gpu::DataFormat::Uint32;
                chain.num_components = 1;
                chain.img_dim = 1;
                chain.width = kWidth;
                chain.height = kHeight;
                chain.depth = 1;
                chain.tile_mode = kTile;
                chain.declared_mip_levels = kMaxMip + 1;
                chain.mip_chain_element_width = kWidth;
                chain.mip_chain_element_height = kHeight;
                chain.mip_chain_bytes_per_block = kBytesPerBlock;
                chain.mip_chain_max_level = kMaxMip;
                chain.mip_chain_base_level = 0;
                chain.gpu_addr = 0x2026900000ull + level0.byte_offset;

                CHECK(prosper::gpu::shader_resource_compute_mip_chain_levels(chain) == kMaxMip + 1,
                      "tiled 2D chain declares nine levels");
                CHECK(prosper::frontend::compute_binding_mip_chain_materializable(chain, false),
                      "valid non-renderer-owned tiled 2D chain is materializable");
                // Live render target with multi-level chain cannot be materialized by compute backend (#3290)
                CHECK(!prosper::frontend::compute_binding_mip_chain_materializable(chain, true),
                      "multi-level compute image binding is declined when aliasing a live render target");

                // Caching multi-level sampled textures with allocation-spanning validation (#3291)
                const auto chain_plan = prosper::gpu::shader_resource_mip_chain_plan(chain);
                const auto single_span = prosper::frontend::compute_sampled_cache_span(
                    single_level, 1, 65536, {});
                CHECK(single_span.eligible && single_span.gpu_addr == single_level.gpu_addr &&
                      single_span.guest_bytes == 65536,
                      "single-level sampled texture uses descriptor address and exact need");

                const auto invalid_chain_span = prosper::frontend::compute_sampled_cache_span(
                    chain, kMaxMip + 1, 65536, {});
                CHECK(!invalid_chain_span.eligible,
                      "multi-level sampled texture declines caching without a valid mip chain plan");

                const auto valid_chain_span = prosper::frontend::compute_sampled_cache_span(
                    chain, kMaxMip + 1, 65536, chain_plan);
                CHECK(valid_chain_span.eligible &&
                      valid_chain_span.gpu_addr == 0x2026900000ull &&
                      valid_chain_span.guest_bytes == chain_plan.allocation_bytes &&
                      valid_chain_span.guest_bytes > 65536,
                      "multi-level sampled texture spans complete allocation from level zero offset");
            }
            CHECK(prosper::frontend::live_compute_graphics_import_native_format(
                      DataFormat::Float32, 1) == 100u && // VK_FORMAT_R32_SFLOAT
                      prosper::frontend::live_compute_graphics_import_native_format(
                          DataFormat::Unorm8, 1) == 9u && // VK_FORMAT_R8_UNORM
                      prosper::frontend::live_compute_graphics_import_native_format(
                          DataFormat::Unorm8, 4) == 37u && // VK_FORMAT_R8G8B8A8_UNORM
                      prosper::frontend::live_compute_graphics_import_native_format(
                          DataFormat::Float16, 1) == 76u && // VK_FORMAT_R16_SFLOAT
                      prosper::frontend::live_compute_graphics_import_native_format(
                          DataFormat::Float16, 2) == 83u && // VK_FORMAT_R16G16_SFLOAT
                      prosper::frontend::live_compute_graphics_import_native_format(
                          DataFormat::Float16, 4) == 97u && // VK_FORMAT_R16G16B16A16_SFLOAT
                      prosper::frontend::live_compute_graphics_import_native_format(
                          DataFormat::Float10_11_11, 3) == 122u &&
                      prosper::frontend::live_compute_graphics_import_native_format(
                          DataFormat::Unorm16, 1) == 70u && // VK_FORMAT_R16_UNORM
                      prosper::frontend::live_compute_graphics_import_native_format(
                          DataFormat::Uint8, 4) == 41u &&
                      prosper::frontend::live_compute_graphics_import_native_format(
                          DataFormat::Unorm8, 3) == 0u,
                  "graphics import policy includes exact FP16/RGBA8 post-FX outputs and rejects unsupported views");
            static const uint32_t transfer_fill_2d_array[] = {
                0x7E080300u,             // v4 = x coordinate
                0x7E0A0280u,             // v5 = y = 0
                0x7E0C0280u,             // v6 = array layer 0
                0x7E0002F2u,             // v0 = 1.0f
                0x7E020280u,             // v1 = 0.0f
                0x7E0402F2u,             // v2 = 1.0f
                0x7E060280u,             // v3 = 0.0f
                0xF0200F28u, 0x00020004u,// IMAGE_STORE RGBA at (v4,v5,v6), DIM=2D_ARRAY
                0xBF810000u,
            };
            const size_t transfer_guest_bytes = W * sizeof(float);
            std::vector<uint8_t> transfer_proof_guest(transfer_guest_bytes, 0x37);
            std::vector<uint8_t> transfer_source_guest(transfer_guest_bytes, 0x52);
            std::vector<uint8_t> transfer_copy_guest(transfer_guest_bytes, 0xa9);
            ShaderResource transfer_producer_dst{};
            transfer_producer_dst.cls = ResourceClass::StorageImage;
            // Syberia's producer uses a real one-layer DIM=2D_ARRAY storage instruction, while
            // the consumer reads the byte-identical base slice through an ordinary non-arrayed
            // DIM=2D sampled instruction.
            transfer_producer_dst.img_dim = 5;
            transfer_producer_dst.binding = 5;
            transfer_producer_dst.sgpr_base = 8;
            transfer_producer_dst.format = DataFormat::Float32;
            transfer_producer_dst.num_components = 1;
            transfer_producer_dst.width = W;
            transfer_producer_dst.height = transfer_producer_dst.depth = 1;
            transfer_producer_dst.gpu_addr =
                reinterpret_cast<uint64_t>(transfer_proof_guest.data());
            transfer_producer_dst.size = static_cast<uint32_t>(transfer_guest_bytes);
            ShaderResource transfer_multilayer = transfer_producer_dst;
            transfer_multilayer.depth = 2;
            CHECK(shader_resource_uses_ordinary_2d_image(
                      transfer_producer_dst, true, false, false) &&
                      !shader_resource_uses_ordinary_2d_image(
                          transfer_producer_dst, true, true, false) &&
                      shader_resource_uses_native_2d_storage_image(
                          transfer_producer_dst, true, true, false) &&
                      !shader_resource_uses_ordinary_2d_image(
                          transfer_producer_dst, true, false, true) &&
                      !shader_resource_uses_native_2d_storage_image(
                          transfer_producer_dst, true, true, true) &&
                      !shader_resource_uses_native_2d_storage_image(
                          transfer_multilayer, true, true, false),
                  "single-layer 2D-array native storage requires one reflected array layer");
            ShaderResourceTable transfer_producer_rt;
            transfer_producer_rt.resources.push_back(transfer_producer_dst);
            ComputeShaderConfig transfer_config;
            transfer_config.user_sgprs.resize(16);
            transfer_config.local_x = W;
            transfer_config.local_y = transfer_config.local_z = 1;
            transfer_config.tidig_comp_cnt = 1;
            transfer_config.native_storage_format_support =
                native_storage_format_support_bit(DataFormat::Float32, 1);
            const std::vector<uint32_t> transfer_producer_spirv = recompile_compute(
                transfer_fill_2d_array, std::size(transfer_fill_2d_array),
                &transfer_producer_rt, transfer_config);
            const DescriptorValidationReport transfer_producer_report =
                validate_spirv_descriptor_interface(
                    transfer_producer_spirv, &transfer_producer_rt, 0,
                    SpirvShaderStage::Compute, false);
            const SpirvDescriptorBinding* transfer_producer_binding =
                find_spirv_descriptor_binding(
                    transfer_producer_report, 0, transfer_producer_dst.binding);
            CHECK(!transfer_producer_spirv.empty() && transfer_producer_report.ok() &&
                      transfer_producer_binding && transfer_producer_binding->storage_float &&
                      transfer_producer_binding->image_dim == 1 &&
                      transfer_producer_binding->image_arrayed &&
                      !transfer_producer_binding->image_multisampled,
                  "single-layer 2D-array producer reflects arrayed exact typed storage");
            if (!transfer_producer_spirv.empty() && transfer_producer_report.ok() &&
                transfer_producer_binding && transfer_producer_binding->storage_float) {
                ComputeItem transfer_proof = it;
                transfer_proof.spirv = transfer_producer_spirv;
                transfer_proof.resources =
                    std::make_shared<ShaderResourceTable>(transfer_producer_rt);
                transfer_proof.code_addr = 0x1122f13du;
                CHECK(prosper::frontend::execute_live_compute_items({transfer_proof}),
                      "native Float32x1 arrayed producer proves complete storage coverage");

                transfer_producer_dst.gpu_addr =
                    reinterpret_cast<uint64_t>(transfer_source_guest.data());
                transfer_producer_rt.resources.back() = transfer_producer_dst;
                ComputeItem transfer_producer = transfer_proof;
                transfer_producer.resources =
                    std::make_shared<ShaderResourceTable>(transfer_producer_rt);
                transfer_producer.dispatch_index = 41;
                transfer_producer.command_order = 10;

                ShaderResourceTable transfer_consumer_rt;
                ShaderResource transfer_sampled = transfer_producer_dst;
                transfer_sampled.cls = ResourceClass::Texture;
                transfer_sampled.binding = 4;
                transfer_sampled.sgpr_base = 0;
                transfer_sampled.swizzle[0] = 4;
                transfer_sampled.swizzle[1] = 5;
                transfer_sampled.swizzle[2] = 6;
                transfer_sampled.swizzle[3] = 7;
                transfer_consumer_rt.resources.push_back(transfer_sampled);
                ShaderResource transfer_copy_dst = transfer_producer_dst;
                transfer_copy_dst.gpu_addr =
                    reinterpret_cast<uint64_t>(transfer_copy_guest.data());
                transfer_consumer_rt.resources.push_back(transfer_copy_dst);
                const std::vector<uint32_t> transfer_consumer_spirv = recompile_compute(
                    image_copy_2d, std::size(image_copy_2d),
                    &transfer_consumer_rt, transfer_config);
                const DescriptorValidationReport transfer_consumer_report =
                    validate_spirv_descriptor_interface(
                        transfer_consumer_spirv, &transfer_consumer_rt, 0,
                        SpirvShaderStage::Compute, false);
                const SpirvDescriptorBinding* transfer_sampled_binding =
                    find_spirv_descriptor_binding(
                        transfer_consumer_report, 0, transfer_sampled.binding);
                const SpirvDescriptorBinding* transfer_copy_binding =
                    find_spirv_descriptor_binding(
                        transfer_consumer_report, 0, transfer_copy_dst.binding);
                CHECK(!transfer_consumer_spirv.empty() && transfer_consumer_report.ok() &&
                          transfer_sampled_binding && transfer_copy_binding &&
                          transfer_sampled_binding->image_dim == 1 &&
                          !transfer_sampled_binding->image_arrayed &&
                          !transfer_sampled_binding->image_multisampled &&
                          transfer_copy_binding->storage_float &&
                          transfer_copy_binding->image_dim == 1 &&
                          !transfer_copy_binding->image_arrayed &&
                          !transfer_copy_binding->image_multisampled,
                      "single-layer 2D-array consumer reflects ordinary sampled/storage views");
                if (!transfer_consumer_spirv.empty() && transfer_consumer_report.ok() &&
                    transfer_sampled_binding && transfer_copy_binding) {
                    ComputeItem transfer_consumer = transfer_proof;
                    transfer_consumer.spirv = transfer_consumer_spirv;
                    transfer_consumer.resources =
                        std::make_shared<ShaderResourceTable>(transfer_consumer_rt);
                    transfer_consumer.code_addr = 0x1122f14du;
                    transfer_consumer.dispatch_index = 42;
                    transfer_consumer.command_order = 20;
                    const uint64_t transfer_seeds_before =
                        prosper::frontend::live_compute_storage_transfer_seeds();
                    const OrderedSubmitResult transfer_result = execute_ordered_items(
                        {{SubmitOperationKind::Dispatch, transfer_producer.dispatch_index,
                          transfer_producer.command_order},
                         {SubmitOperationKind::Dispatch, transfer_consumer.dispatch_index,
                          transfer_consumer.command_order}},
                        {}, {transfer_producer, transfer_consumer},
                        [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                            return RenderedFrame{};
                        },
                        [&](const std::vector<ComputeItem>& items) {
                            return prosper::frontend::execute_live_compute_items(items);
                        },
                        1, 1);
                    const uint64_t transfer_seeds_after =
                        prosper::frontend::live_compute_storage_transfer_seeds();
                    CHECK(transfer_result.compute_executed &&
                              transfer_copy_guest == transfer_source_guest,
                          "arrayed producer to ordinary sampled consumer preserves every Float32 texel");
                    CHECK(!native_2d_compute_transfer_available ||
                              transfer_seeds_after > transfer_seeds_before,
                          "single-layer arrayed native storage producer seeds ordinary 2D "
                          "sampled consumer on-GPU");
                    CHECK(native_2d_compute_transfer_available ||
                              transfer_seeds_after == transfer_seeds_before,
                          "disabled native 2D transfer or authority validation keeps the exact "
                          "guest fallback");
                }
            }

#if defined(__linux__)
            // Use a dedicated mapping so page protection cannot alias unrelated malloc metadata.
            // Outside an ordered submit there is no submit-local journal, so a successful import
            // here specifically proves the cross-submit write-watch authority path.
            constexpr size_t import_mapping_bytes = 4096;
            auto* import_guest = static_cast<uint8_t*>(mmap(
                nullptr, import_mapping_bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            CHECK(import_guest != MAP_FAILED,
                  "allocate dedicated typed-storage import mapping");
            if (import_guest != MAP_FAILED) {
                std::memset(import_guest, 0x37, import_mapping_bytes);
                prosper::host::guest_write_watch_set_fault_onstack(true);
                prosper::host::guest_write_watch_notify_direct_mapping_added(
                    reinterpret_cast<uint64_t>(import_guest), import_mapping_bytes,
                    0x7e0000, 0x3 /* SCE CPU_READ|CPU_WRITE */);
                ShaderResourceTable import_rt = fill_rt;
                ShaderResource& import_dst = import_rt.resources.back();
                import_dst.gpu_addr = reinterpret_cast<uint64_t>(import_guest);
                ComputeItem import_item = it;
                import_item.resources = std::make_shared<ShaderResourceTable>(import_rt);
                CHECK(prosper::frontend::execute_live_compute_items({import_item}) &&
                          prosper::frontend::execute_live_compute_items({import_item}) &&
                          prosper::frontend::execute_live_compute_items({import_item}) &&
                          prosper::frontend::execute_live_compute_items({import_item}),
                      "repeated typed-storage output promotes its exact source watch");
                ShaderResource sampled_fill = import_dst;
                sampled_fill.cls = ResourceClass::Texture;
                prosper::frontend::LiveComputeImageImport compute_import;
                CHECK(prosper::frontend::import_live_compute_storage_image(
                          sampled_fill, import_dst.size, compute_import) &&
                          compute_import.valid() && compute_import.width == W &&
                          compute_import.height == 1 && compute_import.depth == 1 &&
                          compute_import.native_format && compute_import.layout,
                      "validated typed-storage result can be leased by an exact sampled descriptor");
                prosper::frontend::LiveComputeImageImport rejected_import;
                ShaderResource mismatched_fill = sampled_fill;
                ++mismatched_fill.width;
                const bool rejected_extent =
                    !prosper::frontend::import_live_compute_storage_image(
                        mismatched_fill, import_dst.size, rejected_import);
                mismatched_fill = sampled_fill;
                ++mismatched_fill.tile_mode;
                const bool rejected_layout =
                    !prosper::frontend::import_live_compute_storage_image(
                        mismatched_fill, import_dst.size, rejected_import);
                mismatched_fill = sampled_fill;
                mismatched_fill.host_data = import_guest;
                mismatched_fill.host_data_size = import_dst.size;
                const bool rejected_replay =
                    !prosper::frontend::import_live_compute_storage_image(
                        mismatched_fill, import_dst.size, rejected_import);
                mismatched_fill = sampled_fill;
                mismatched_fill.srgb = true;
                const bool rejected_srgb =
                    !prosper::frontend::import_live_compute_storage_image(
                        mismatched_fill, import_dst.size, rejected_import);
                mismatched_fill = sampled_fill;
                mismatched_fill.declared_mip_levels = 2;
                const bool rejected_mips =
                    !prosper::frontend::import_live_compute_storage_image(
                        mismatched_fill, import_dst.size, rejected_import);
                CHECK(rejected_extent && rejected_layout && rejected_replay && rejected_srgb &&
                          rejected_mips,
                      "compute-image import requires full identity and keeps replay/mip/sRGB fallback");
                compute_import = {};
                // This binary does not install the emulator's SIGSEGV write-fault handler. Mark the
                // same page dirty through the DMA/GPU side of the watch API; the importer must not
                // trust the retained image after any architectural writer invalidates its bytes.
                prosper::host::guest_write_watch_notify_gpu_write(
                    reinterpret_cast<uint64_t>(import_guest), import_dst.size);
                CHECK(!prosper::frontend::import_live_compute_storage_image(
                          sampled_fill, import_dst.size, compute_import),
                      "guest mutation revokes a cross-submit compute-image import");
                prosper::host::guest_write_watch_notify_direct_mapping_removed(
                    reinterpret_cast<uint64_t>(import_guest), import_mapping_bytes);
                prosper::host::guest_write_watch_set_fault_onstack(false);
                munmap(import_guest, import_mapping_bytes);
            }
#endif

            // Prove a second full writer of the same format/extent on a different target, then use
            // it on this still-valid guest mirror. The persistent image key intentionally does not
            // contain the shader: different post-processes may reuse one target. Its GPU comparison
            // must report the changed word exactly and force the ordinary writeback path.
            static const uint32_t changed_fill_1d[] = {
                0x7E080300u,             // v4 = v0 (x)
                0x7E0A0301u,             // v5 = v1 (y)
                0x7E0402F2u,             // v2 = 1.0f (changed stored B)
                0x7E060280u,             // v3 = 0
                0xF0200F08u, 0x00020004u,// IMAGE_STORE v0..v3 at (v4,v5)
                0xBF810000u,
            };
            std::vector<uint8_t> changed_proof_guest(W * 8, 0x71);
            ShaderResourceTable changed_proof_rt = fill_rt;
            changed_proof_rt.resources.back().gpu_addr =
                reinterpret_cast<uint64_t>(changed_proof_guest.data());
            std::vector<uint32_t> changed_spirv = recompile_compute(
                changed_fill_1d,
                sizeof(changed_fill_1d) / sizeof(changed_fill_1d[0]),
                &changed_proof_rt, fill_config);
            CHECK(!changed_spirv.empty(), "second full-coverage fill kernel recompiles");
            if (!changed_spirv.empty()) {
                ComputeItem changed = it;
                changed.spirv = changed_spirv;
                changed.code_addr = 0x1122f12du;
                changed.resources = std::make_shared<ShaderResourceTable>(changed_proof_rt);
                CHECK(prosper::frontend::execute_live_compute_items({changed}),
                      "changed fill proves full coverage on an independent target");
                const std::vector<uint8_t> changed_expected = changed_proof_guest;
                changed.resources = std::make_shared<ShaderResourceTable>(fill_rt);

                prosper::frontend::live_compute_fail_next_storage_readback_for_test();
                CHECK(!prosper::frontend::execute_live_compute_items({changed}),
                      "injected post-submit storage readback failure is reported");
                CHECK(fill_equals(after_prove),
                      "failed storage readback does not publish newer image bytes to the guest");
                uint32_t changed_write_notifications = 0;
                set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char*) {
                    if (addr == fdst.gpu_addr && size == fdst.size)
                        ++changed_write_notifications;
                });
#if defined(__linux__)
                // The nested compute-import fixture temporarily disables the synthetic fault
                // handler when it tears down. Restore the outer mapping's emulator contract before
                // exercising this target again.
                prosper::host::guest_write_watch_set_fault_onstack(true);
                const uint64_t storage_snapshots_before =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
#endif
                CHECK(prosper::frontend::execute_live_compute_items({changed}),
                      "storage image retries after a failed post-submit readback");
                set_guest_gpu_write_observer({});
                CHECK(fill_equals(changed_expected) && !fill_equals(after_prove),
                      "retry invalidates the stale baseline and publishes the changed output");
                CHECK(changed_write_notifications == 1,
                      "recovered storage output forces guest writeback and invalidation");
#if defined(__linux__)
                const uint64_t storage_snapshots_after =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
                CHECK(storage_snapshots_after >=
                          storage_snapshots_before + fill_guest_bytes,
                      "post-failure repair replaces its invalidated exact result baseline");

                auto recovered_write_watch = prosper::host::GuestWriteWatch::create(
                    reinterpret_cast<uint64_t>(fill_guest), fill_guest_bytes);
                const uint64_t recovered_snapshots_before =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
#endif
                uint32_t recovered_repeat_notifications = 0;
                set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char*) {
                    if (addr == fdst.gpu_addr && size == fdst.size)
                        ++recovered_repeat_notifications;
                });
                CHECK(prosper::frontend::execute_live_compute_items({changed}),
                      "recovered storage image repeats an identical dispatch");
                set_guest_gpu_write_observer({});
                CHECK(fill_equals(changed_expected),
                      "post-recovery identical dispatch preserves the repaired guest result");
                CHECK(recovered_repeat_notifications == 1,
                      "post-recovery identical result still invalidates renderer aliases");
#if defined(__linux__)
                CHECK(static_cast<bool>(recovered_write_watch) &&
                          recovered_write_watch.query() ==
                              prosper::host::GuestWriteWatchQuery::Unchanged,
                      "post-recovery identical result does not rewrite guest bytes");
                CHECK(prosper::frontend::live_compute_storage_result_snapshot_bytes() ==
                          recovered_snapshots_before,
                      "post-recovery identical result does not recopy its repaired baseline");
                recovered_write_watch.reset();
#endif
            }

            // Guest memory no longer matches the retained source. Exact source validation must
            // force a writeback rather than trusting only the GPU-side output baseline.
#if defined(__linux__)
            // The production emulator installs the SIGSEGV write-fault handler; this compact test
            // does not. Model its architectural CPU write through the ordinary pre-write hook so
            // the armed page is made writable and marked Dirty before touching it.
            prosper::host::guest_write_watch_notify_host_write(
                reinterpret_cast<uint64_t>(fill_guest), 1);
#endif
            fill_guest[0] ^= 0xff;
            uint32_t repaired_write_notifications = 0;
            set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char*) {
                if (addr == fdst.gpu_addr && size == fdst.size)
                    ++repaired_write_notifications;
            });
            CHECK(prosper::frontend::execute_live_compute_items({it}),
                  "externally changed guest mirror reruns the retained full-coverage dispatch");
            set_guest_gpu_write_observer({});
            CHECK(fill_equals(after_prove),
                  "retained output repairs an externally changed guest mirror");
            CHECK(repaired_write_notifications == 1,
                  "externally changed guest mirror forces writeback and invalidation");

#if defined(__linux__)
            // Alternate two proven-full writers and verify that a changing target carries no exact
            // source snapshot: its old seed is unobservable, and the next GPU result comparison is
            // independently collision-free.
            if (!changed_spirv.empty()) {
                ComputeItem alternating_changed = it;
                alternating_changed.spirv = changed_spirv;
                alternating_changed.code_addr = 0x1122f12du;
                alternating_changed.resources = std::make_shared<ShaderResourceTable>(fill_rt);
                const uint64_t dynamic_snapshots_before =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
                ShaderResource dynamic_sampled = fdst;
                dynamic_sampled.cls = ResourceClass::Texture;
                prosper::frontend::LiveComputeImageImport dynamic_import;
                CHECK(prosper::frontend::import_live_compute_storage_image(
                          dynamic_sampled, fill_guest_bytes, dynamic_import) &&
                          dynamic_import.valid(),
                      "warm changing target has cross-submit watch-backed export authority");
                dynamic_import = {};
                const auto dynamic_watch_before = prosper::host::guest_write_watch_stats();
                CHECK(prosper::frontend::execute_live_compute_items({alternating_changed}) &&
                          prosper::frontend::execute_live_compute_items({it}),
                      "alternating full-coverage storage results execute");
                const auto dynamic_watch_after = prosper::host::guest_write_watch_stats();
                CHECK(dynamic_watch_after.create_attempts == dynamic_watch_before.create_attempts,
                      "warm changing full writers reuse their export watch registration");
                CHECK(dynamic_watch_after.rearms >= dynamic_watch_before.rearms + 2,
                      "each changing writer rearms its retained export watch at publication");
                CHECK(fill_equals(after_prove),
                      "watch retention preserves exact alternating storage results");
                CHECK(prosper::frontend::import_live_compute_storage_image(
                          dynamic_sampled, fill_guest_bytes, dynamic_import) &&
                          dynamic_import.valid(),
                      "retained watch authorizes the completed result across submits");
                dynamic_import = {};
                const uint64_t dynamic_snapshots_after =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
                if (adaptive_storage_result_validation_enabled) {
                    CHECK(dynamic_snapshots_after == dynamic_snapshots_before,
                          "dynamic full-overwrite result avoids redundant snapshots");
                } else {
                    CHECK(dynamic_snapshots_after >=
                              dynamic_snapshots_before + 2 * fill_guest_bytes,
                          "disabled adaptive policy preserves exact dynamic snapshots");
                }
                // The installed alternate-stack handler must dirty the retained registration on
                // a real guest store, without a submit-local journal or an explicit GPU notification.
                fill_guest[0] ^= 0xff;
                CHECK(!prosper::frontend::import_live_compute_storage_image(
                          dynamic_sampled, fill_guest_bytes, dynamic_import),
                      "CPU write revokes a reused cross-submit export watch");
                CHECK(prosper::frontend::execute_live_compute_items({it}) &&
                          fill_equals(after_prove),
                      "ordinary writeback repairs a guest store after watch reuse");
                prosper::host::guest_write_watch_notify_direct_mapping_removed(
                    reinterpret_cast<uint64_t>(fill_guest), fill_mapping_bytes);
                CHECK(mmap(fill_guest, fill_mapping_bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == fill_guest,
                      "replace warm storage target backing at the same virtual address");
                prosper::host::guest_write_watch_notify_direct_mapping_added(
                    reinterpret_cast<uint64_t>(fill_guest), fill_mapping_bytes,
                    0x33860000, 0x3 /* SCE CPU_READ|CPU_WRITE */);
                CHECK(prosper::frontend::execute_live_compute_items({alternating_changed}) &&
                          prosper::frontend::execute_live_compute_items({it}) &&
                          fill_equals(after_prove),
                      "full writers rebuild authority after guest target remapping");
                CHECK(prosper::frontend::import_live_compute_storage_image(
                          dynamic_sampled, fill_guest_bytes, dynamic_import) &&
                          dynamic_import.valid(),
                      "replacement mapping obtains fresh cross-submit export authority");
                dynamic_import = {};
                fill_guest[0] ^= 0xff;
                CHECK(!prosper::frontend::import_live_compute_storage_image(
                          dynamic_sampled, fill_guest_bytes, dynamic_import),
                      "CPU store to remapped target revokes the recreated export watch");
                CHECK(prosper::frontend::execute_live_compute_items({it}) &&
                          fill_equals(after_prove),
                      "ordinary writeback repairs the remapped target after its CPU store");
            }
#endif

            // The first dispatch to a new target is the cache-churn shape seen in full-resolution
            // post-processing: the shader/extent already has a Full proof, but this address has no
            // retained image yet. A dedicated invocation lowers the production crossover to zero so
            // this reduced target exercises the real deferring branch without a 64 MiB allocation;
            // the ordinary and disable-switch invocations retain the previous immediate baseline.
            std::vector<uint8_t> cold_guest(W * 8, 0x6b);
            ShaderResourceTable cold_rt = fill_rt;
            cold_rt.resources.back().gpu_addr =
                reinterpret_cast<uint64_t>(cold_guest.data());
            ComputeItem cold_item = it;
            cold_item.resources = std::make_shared<ShaderResourceTable>(cold_rt);
            const uint64_t cold_source_snapshots_before =
                prosper::frontend::live_compute_storage_result_snapshot_bytes();
            const uint64_t cold_result_snapshots_before =
                prosper::frontend::live_compute_image_result_snapshot_bytes();
            prosper::frontend::live_compute_zero_next_cold_storage_snapshot_minimum_for_test();
            CHECK(prosper::frontend::execute_live_compute_items({cold_item}),
                  "proven-full writer executes on a cold retained target");
            const uint64_t cold_source_snapshots_after =
                prosper::frontend::live_compute_storage_result_snapshot_bytes();
            const uint64_t cold_result_snapshots_after =
                prosper::frontend::live_compute_image_result_snapshot_bytes();
            if (cold_storage_snapshot_deferral_enabled) {
#if defined(_WIN32)
                CHECK(cold_source_snapshots_after >=
                          cold_source_snapshots_before + fill_guest_bytes,
                      "Windows retains an exact guest mirror for a cold exportable target");
#else
                CHECK(cold_source_snapshots_after == cold_source_snapshots_before,
                      "deferring policy admits a cold proven-full target without a source copy");
#endif

                // A later standalone backend invocation has no same-submit journal authority. The
                // first repeat uses either Windows' exact mirror or the platform write-watch/
                // deferred-source path. Both remain collision-free and preserve the guest result.
                const std::vector<uint8_t> cold_expected = cold_guest;
                const uint64_t repeat_snapshots_before =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
                CHECK(prosper::frontend::execute_live_compute_items({cold_item}),
                      "first invalidated repeat repairs a deferred cold target");
#if defined(_WIN32)
                CHECK(cold_guest == cold_expected &&
                          prosper::frontend::live_compute_storage_result_snapshot_bytes() ==
                              repeat_snapshots_before,
                      "Windows exact mirror authorizes the repeat without recopying its baseline");
#else
                CHECK(cold_guest == cold_expected &&
                          prosper::frontend::live_compute_storage_result_snapshot_bytes() >=
                              repeat_snapshots_before + fill_guest_bytes,
                      "first invalidated repeat establishes exact source authority");
#endif

                cold_guest[0] ^= 0xff;
                CHECK(cold_guest != cold_expected,
                      "external mutation changes the deferred target fixture");
                CHECK(prosper::frontend::execute_live_compute_items({cold_item}) &&
                          cold_guest == cold_expected,
                      "external mutation forces deferred target writeback and exact repair");
            } else {
                CHECK(cold_source_snapshots_after >=
                          cold_source_snapshots_before + fill_guest_bytes,
                      "default or disabled policy retains the cold source immediately");
            }
            CHECK(cold_result_snapshots_after == cold_result_snapshots_before,
                  "cold exact target retains its GPU baseline without a transient host copy");
            CHECK(prosper::frontend::cold_storage_result_snapshot_can_defer(
                      false, true, 64u << 20, 16u << 20) &&
                      !prosper::frontend::cold_storage_result_snapshot_can_defer(
                          true, true, 64u << 20, 16u << 20) &&
                      !prosper::frontend::cold_storage_result_snapshot_can_defer(
                          false, false, 64u << 20, 16u << 20) &&
                      !prosper::frontend::cold_storage_result_snapshot_can_defer(
                          false, true, 8u << 20, 16u << 20),
                  "cold snapshot deferral is limited to large proven-full guest targets");

            // Model the exact stale-fallback sequence from the review: transient setup stores host
            // result A, result B reaches guest memory but GPU-baseline ownership fails, then A runs
            // again. The failed B ownership attempt must invalidate host A; otherwise the final A
            // can be misclassified as repeated and leave B in architectural guest memory.
            if (!changed_spirv.empty()) {
                std::vector<uint8_t> fallback_guest(W * 8, 0x42);
                ShaderResourceTable fallback_rt = fill_rt;
                fallback_rt.resources.back().gpu_addr =
                    reinterpret_cast<uint64_t>(fallback_guest.data());
                ComputeItem fallback_a = it;
                fallback_a.resources = std::make_shared<ShaderResourceTable>(fallback_rt);
                ComputeItem fallback_b = fallback_a;
                fallback_b.spirv = changed_spirv;
                fallback_b.code_addr = 0x1122f12du;
                fallback_b.dispatch_index = 51;
                fallback_b.command_order = 10;
                fallback_a.dispatch_index = 52;
                fallback_a.command_order = 20;

                const uint64_t fallback_snapshots_before =
                    prosper::frontend::live_compute_image_result_snapshot_bytes();
                prosper::frontend::live_compute_force_next_image_result_host_fallback_for_test();
                CHECK(prosper::frontend::execute_live_compute_items({fallback_a}),
                      "transient baseline setup fallback publishes result A");
                const uint64_t fallback_snapshots_after_a =
                    prosper::frontend::live_compute_image_result_snapshot_bytes();
                CHECK(std::equal(fallback_guest.begin(), fallback_guest.end(),
                                 after_prove.begin()) &&
                          fallback_snapshots_after_a >=
                              fallback_snapshots_before + fill_guest_bytes,
                      "transient setup fallback retains exact host result A");

                prosper::frontend::live_compute_fail_next_image_result_buffer_retain_for_test();
                size_t fallback_dispatches = 0;
                bool fallback_dispatches_ok = true;
                bool failed_retain_published_b = false;
                const OrderedSubmitResult fallback_submit = execute_ordered_items(
                    {{SubmitOperationKind::Dispatch, fallback_b.dispatch_index,
                      fallback_b.command_order},
                     {SubmitOperationKind::Dispatch, fallback_a.dispatch_index,
                      fallback_a.command_order}},
                    {}, {fallback_b, fallback_a},
                    [](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                        return RenderedFrame{};
                    },
                    [&](const std::vector<ComputeItem>& items) {
                        const bool ok = prosper::frontend::execute_live_compute_items(items);
                        fallback_dispatches_ok &= ok;
                        if (fallback_dispatches++ == 0)
                            failed_retain_published_b = std::equal(
                                fallback_guest.begin(), fallback_guest.end(),
                                changed_proof_guest.begin()) &&
                                prosper::frontend::live_compute_image_result_snapshot_bytes() ==
                                    fallback_snapshots_after_a;
                        return ok;
                    },
                    1, 1);
                CHECK(fallback_submit.compute_executed && fallback_dispatches_ok &&
                          fallback_dispatches == 2 && failed_retain_published_b,
                      "result B survives injected GPU-baseline ownership failure");
                CHECK(std::equal(fallback_guest.begin(), fallback_guest.end(),
                                 after_prove.begin()),
                      "stale host result A cannot suppress required A-after-B writeback");
            }
        }
#if defined(__linux__)
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(fill_guest), fill_mapping_bytes);
        prosper::host::guest_write_watch_set_fault_onstack(false);
        munmap(fill_guest, fill_mapping_bytes);
#endif
    }

    // A proven-full write-only raw storage image may retain its RGBA32_UINT interchange result even
    // though that representation is not canonical guest data: the old raw texels are unobservable.
    // This is the Plucky Squire lighting-grid shape (3D tiled FP16), reduced to one 4x4x4 workgroup.
    // The third identical dispatch must compare the raw transfer on-GPU and omit pack/retile and the
    // guest invalidation, while an external guest-memory change must still force exact writeback.
    {
        static const uint32_t fill_3d[] = {
            0x7E080300u,             // v4 = v0 (x)
            0x7E0A0301u,             // v5 = v1 (y)
            0x7E0C0302u,             // v6 = v2 (z)
            0xF0200F10u, 0x00020004u,// IMAGE_STORE v0..v3 at (v4,v5,v6), dim:3D, no load
            0xBF810000u,
        };
        constexpr uint32_t RW = 4, RH = 4, RD = 4;
        constexpr uint32_t raw_mode = static_cast<uint32_t>(TileMode::Sw64KbS);
        const size_t raw_guest_bytes = tiled_volume_bytes(RW, RH, RD, raw_mode, 8);
        std::vector<uint8_t> raw_guest_storage;
        uint8_t* raw_guest = nullptr;
#if defined(__linux__)
        void* raw_mapping = mmap(nullptr, raw_guest_bytes, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(raw_mapping != MAP_FAILED, "map raw storage-image regression range");
        if (raw_mapping == MAP_FAILED) return fails ? fails : 1;
        raw_guest = static_cast<uint8_t*>(raw_mapping);
        prosper::host::guest_write_watch_set_fault_onstack(true);
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(raw_guest), raw_guest_bytes, 0x7d0000,
            0x3 /* SCE CPU_READ|CPU_WRITE */);
#else
        raw_guest_storage.resize(raw_guest_bytes);
        raw_guest = raw_guest_storage.data();
#endif
        std::fill_n(raw_guest, raw_guest_bytes, 0x9d);
        ShaderResourceTable raw_rt;
        ShaderResource raw_dst{};
        raw_dst.cls = ResourceClass::StorageImage;
        raw_dst.img_dim = 2;
        raw_dst.binding = 5;
        raw_dst.sgpr_base = 8;
        raw_dst.format = DataFormat::Float16;
        raw_dst.num_components = 4;
        raw_dst.width = RW;
        raw_dst.height = RH;
        raw_dst.depth = RD;
        raw_dst.tile_mode = raw_mode;
        raw_dst.gpu_addr = reinterpret_cast<uint64_t>(raw_guest);
        raw_dst.size = static_cast<uint32_t>(raw_guest_bytes);
        raw_rt.resources.push_back(raw_dst);
        ComputeShaderConfig raw_config;
        raw_config.user_sgprs.resize(16);
        raw_config.local_x = RW;
        raw_config.local_y = RH;
        raw_config.local_z = RD;
        raw_config.tidig_comp_cnt = 2;
        // Keep the raw fallback used by true 3D images even on devices supporting typed FP16.
        raw_config.native_storage_format_support = 0;
        const std::vector<uint32_t> raw_spirv = recompile_compute(
            fill_3d, std::size(fill_3d), &raw_rt, raw_config);
        CHECK(!raw_spirv.empty(), "write-only raw 3D fill kernel recompiles");
        if (!raw_spirv.empty()) {
            ComputeItem raw_item;
            raw_item.spirv = raw_spirv;
            raw_item.resources = std::make_shared<ShaderResourceTable>(raw_rt);
            raw_item.launch.threads_x = RW;
            raw_item.launch.threads_y = RH;
            raw_item.launch.threads_z = RD;
            raw_item.launch.local_x = RW;
            raw_item.launch.local_y = RH;
            raw_item.launch.local_z = RD;
            raw_item.launch.groups_x = raw_item.launch.groups_y = raw_item.launch.groups_z = 1;
            raw_item.code_addr = 0x1122f13du;
            CHECK(prosper::frontend::execute_live_compute_items({raw_item}),
                  "raw 3D fill proves complete write coverage");
            const std::vector<uint8_t> raw_expected(raw_guest,
                                                    raw_guest + raw_guest_bytes);
            CHECK(prosper::frontend::execute_live_compute_items({raw_item}),
                  "raw 3D fill establishes a retained exact result baseline");
#if defined(__linux__)
            auto raw_repeat_watch = prosper::host::GuestWriteWatch::create(
                reinterpret_cast<uint64_t>(raw_guest), raw_guest_bytes);
#endif
            uint32_t raw_repeat_notifications = 0;
            set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char*) {
                if (addr == raw_dst.gpu_addr && size == raw_guest_bytes)
                    ++raw_repeat_notifications;
            });
            CHECK(prosper::frontend::execute_live_compute_items({raw_item}),
                  "retained raw 3D fill repeats an identical dispatch");
            set_guest_gpu_write_observer({});
            CHECK(std::equal(raw_guest, raw_guest + raw_guest_bytes, raw_expected.begin()) &&
                      raw_repeat_notifications == 1,
                  "GPU-identical raw 3D output skips pack/retile but invalidates renderer aliases");
#if defined(__linux__)
            CHECK(static_cast<bool>(raw_repeat_watch) && raw_repeat_watch.query() ==
                      prosper::host::GuestWriteWatchQuery::Unchanged,
                  "GPU-identical raw 3D output leaves guest-byte dirty tracking clean");
            raw_repeat_watch.reset();
#endif

            raw_guest[0] ^= 0xff;
            uint32_t raw_repair_notifications = 0;
            set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char*) {
                if (addr == raw_dst.gpu_addr && size == raw_guest_bytes)
                    ++raw_repair_notifications;
            });
            CHECK(prosper::frontend::execute_live_compute_items({raw_item}),
                  "externally changed raw 3D mirror reruns the retained dispatch");
            set_guest_gpu_write_observer({});
            CHECK(std::equal(raw_guest, raw_guest + raw_guest_bytes, raw_expected.begin()) &&
                      raw_repair_notifications == 1,
                  "external raw 3D mirror change forces exact writeback and invalidation");
        }

        // The same 3D FP16 resource can stay at its exact eight-byte native width when the device
        // advertises dimension-specific support. This is Astro Bot's 240x135x64 ping-pong shape;
        // reflection must select float storage rather than the sixteen-byte raw-uvec4 interchange.
        ComputeShaderConfig native_3d_config = raw_config;
        native_3d_config.native_storage_format_support =
            native_storage_3d_format_support_bit(DataFormat::Float16, 4);
        const std::vector<uint32_t> native_3d_spirv = recompile_compute(
            fill_3d, std::size(fill_3d), &raw_rt, native_3d_config);
        CHECK(!native_3d_spirv.empty(), "native typed 3D fill kernel recompiles");
        if (!native_3d_spirv.empty()) {
            const DescriptorValidationReport native_3d_report =
                validate_spirv_descriptor_interface(
                    native_3d_spirv, &raw_rt, 0, SpirvShaderStage::Compute, false);
            const SpirvDescriptorBinding* native_3d_binding =
                find_spirv_descriptor_binding(native_3d_report, 0, raw_dst.binding);
            CHECK(native_3d_report.ok() && native_3d_binding &&
                      native_3d_binding->kind == SpirvDescriptorKind::StorageImage &&
                      native_3d_binding->storage_float && native_3d_binding->image_dim == 2,
                  "dimension-capable FP16 volume reflects exact float 3D storage");

#if defined(__linux__)
            const bool native_3d_runtime_supported =
                prosper::frontend::live_compute_native_storage_3d_supported(
                    DataFormat::Float16, 4, RW, RH, RD);
#else
            constexpr bool native_3d_runtime_supported = false;
#endif
            if (native_3d_runtime_supported &&
                adaptive_storage_result_validation_enabled) {
#if defined(__linux__)
                prosper::host::guest_write_watch_notify_host_write(
                    reinterpret_cast<uint64_t>(raw_guest), raw_guest_bytes);
#endif
                std::fill_n(raw_guest, raw_guest_bytes, 0x3a);
                ComputeItem native_3d_item;
                native_3d_item.spirv = native_3d_spirv;
                native_3d_item.resources = std::make_shared<ShaderResourceTable>(raw_rt);
                native_3d_item.launch.threads_x = RW;
                native_3d_item.launch.threads_y = RH;
                native_3d_item.launch.threads_z = RD;
                native_3d_item.launch.local_x = RW;
                native_3d_item.launch.local_y = RH;
                native_3d_item.launch.local_z = RD;
                native_3d_item.launch.groups_x = native_3d_item.launch.groups_y =
                    native_3d_item.launch.groups_z = 1;
                native_3d_item.code_addr = 0x1122f14du;
                CHECK(prosper::frontend::execute_live_compute_items({native_3d_item}),
                      "native typed 3D fill executes at exact guest width");
                const std::vector<uint8_t> native_3d_expected(
                    raw_guest, raw_guest + raw_guest_bytes);
                CHECK(std::any_of(native_3d_expected.begin(), native_3d_expected.end(),
                                  [](uint8_t byte) { return byte != 0x3a; }),
                      "native typed 3D fill overwrites its guest volume");
                CHECK(prosper::frontend::execute_live_compute_items({native_3d_item}) &&
                          std::equal(raw_guest, raw_guest + raw_guest_bytes,
                                     native_3d_expected.begin()),
                      "retained native 3D result is byte-exact across repeated dispatches");
#if defined(__linux__)
                // The first successful dispatch proves full coverage and the second retains the
                // image. Force one exact repair so the retained result establishes the cross-submit
                // page watch that authorizes the following device-local sampled copy.
                raw_guest[0] ^= 0x5a;
                CHECK(prosper::frontend::execute_live_compute_items({native_3d_item}) &&
                          std::equal(raw_guest, raw_guest + raw_guest_bytes,
                                     native_3d_expected.begin()),
                      "native 3D result repairs an external mirror write before transfer");
#endif

                ShaderResource sampled_native_3d = raw_dst;
                sampled_native_3d.cls = ResourceClass::Texture;
                prosper::frontend::LiveComputeImageImport graphics_import;
                CHECK(prosper::frontend::import_live_compute_storage_image(
                          sampled_native_3d, raw_guest_bytes, graphics_import) &&
                          graphics_import.valid() && graphics_import.width == RW &&
                          graphics_import.height == RH && graphics_import.depth == RD,
                      "validated native 3D storage result can be leased by graphics");
                ShaderResource mismatched_native_3d = sampled_native_3d;
                --mismatched_native_3d.depth;
                prosper::frontend::LiveComputeImageImport rejected_graphics_import;
                CHECK(!prosper::frontend::import_live_compute_storage_image(
                          mismatched_native_3d, raw_guest_bytes, rejected_graphics_import),
                      "graphics volume import requires exact 3D descriptor identity");
                graphics_import = {};

                static const uint32_t sampled_copy_3d[] = {
                    0x7E080300u,             // v4 = v0 (x)
                    0x7E0A0301u,             // v5 = v1 (y)
                    0x7E0C0302u,             // v6 = v2 (z)
                    0xF0000F10u, 0x00000004u,// IMAGE_LOAD v0..v3 at (v4,v5,v6), dim:3D
                    0xBF8C3F70u,             // s_waitcnt vmcnt(0)
                    0xF0200F10u, 0x00020004u,// IMAGE_STORE to a distinct 3D destination
                    0xBF810000u,
                };
                std::vector<uint8_t> native_copy_guest(raw_guest_bytes, 0x71);
                ShaderResourceTable native_copy_rt;
                ShaderResource native_copy_src = raw_dst;
                native_copy_src.cls = ResourceClass::Texture;
                native_copy_src.binding = 4;
                native_copy_src.sgpr_base = 0;
                native_copy_rt.resources.push_back(native_copy_src);
                ShaderResource native_copy_dst = raw_dst;
                native_copy_dst.gpu_addr =
                    reinterpret_cast<uint64_t>(native_copy_guest.data());
                native_copy_rt.resources.push_back(native_copy_dst);
                const std::vector<uint32_t> native_copy_spirv = recompile_compute(
                    sampled_copy_3d, std::size(sampled_copy_3d),
                    &native_copy_rt, native_3d_config);
                CHECK(!native_copy_spirv.empty(),
                      "sampled-to-storage native 3D copy kernel recompiles");
                if (!native_copy_spirv.empty()) {
                    ComputeItem native_copy_item = native_3d_item;
                    native_copy_item.spirv = native_copy_spirv;
                    native_copy_item.resources =
                        std::make_shared<ShaderResourceTable>(native_copy_rt);
                    native_copy_item.code_addr = 0x1122f15du;
                    const uint64_t transfer_seeds_before =
                        prosper::frontend::live_compute_storage_transfer_seeds();
                    CHECK(prosper::frontend::execute_live_compute_items({native_copy_item}),
                          "sampled 3D consumer executes from retained native storage output");
                    CHECK(prosper::frontend::live_compute_storage_transfer_seeds() >
                              transfer_seeds_before,
                          "sampled 3D consumer seeds on-GPU without a guest upload");
                    std::vector<uint8_t> native_source_linear(RW * RH * RD * 8u);
                    std::vector<uint8_t> native_copy_linear(native_source_linear.size());
                    CHECK(detile_volume(native_source_linear.data(), raw_guest,
                                        raw_guest_bytes, RW, RH, RD, raw_mode, 8) &&
                              detile_volume(native_copy_linear.data(),
                                            native_copy_guest.data(), raw_guest_bytes,
                                            RW, RH, RD, raw_mode, 8) &&
                              native_copy_linear == native_source_linear,
                          "device-local native 3D seed preserves every logical FP16 voxel");
                }
            }
        }
#if defined(__linux__)
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(raw_guest), raw_guest_bytes);
        prosper::host::guest_write_watch_set_fault_onstack(false);
        munmap(raw_guest, raw_guest_bytes);
#endif
    }

    // Syberia's colour-grade LUT is a tiled R11G11B10F volume written by one compute dispatch and
    // sampled by the immediately following dispatch in the same guest submit (#1790). Existing
    // coverage separates that contract into 2D packed-R11 conversion, 3D FP16 storage, and a 2D
    // same-submit graphics import. Keep the exact missing cross-product together here: a real 3D
    // SAMPLE_LZ consumer, the guest's SW_64KB_R_X layout, exact storage with and without a native
    // typed-format capability bit, and a quantitative f11/f10 oracle over every logical voxel.
    {
        static const uint32_t r11_volume_producer[] = {
            0xF0000F10u, 0x00000400u, // IMAGE_LOAD v[4:7], v[0:2], s[0:7], dim:3D
            0xBF8C3F70u,              // s_waitcnt vmcnt(0)
            0xF0200F10u, 0x00020400u, // IMAGE_STORE v[4:7], v[0:2], s[8:15], dim:3D
            0xBF810000u,
        };
        static const uint32_t r11_volume_consumer[] = {
            0x7E080300u,              // v4 = v0 (integer x retained in v0)
            0x7E0A0301u,              // v5 = v1 (integer y retained in v1)
            0x7E0C0302u,              // v6 = v2 (integer z retained in v2)
            0x7E080D04u,              // v_cvt_f32_u32 v4, v4
            0x7E0A0D05u,              // v_cvt_f32_u32 v5, v5
            0x7E0C0D06u,              // v_cvt_f32_u32 v6, v6
            0x060808F0u,              // v4 = v4 + 0.5
            0x060A0AF0u,              // v5 = v5 + 0.5
            0x060C0CF0u,              // v6 = v6 + 0.5
            0x100808FFu, 0x3E800000u, // v4 *= 0.25 (literal)
            0x100A0AFFu, 0x3E800000u, // v5 *= 0.25 (literal)
            0x100C0CFFu, 0x3E800000u, // v6 *= 0.25 (literal)
            0xF09C0F10u, 0x00400C04u, // IMAGE_SAMPLE_LZ v[12:15], v[4:6], s[0:7], s[8:11], 3D
            0xBF8C3F70u,              // s_waitcnt vmcnt(0)
            0xF0200F10u, 0x00040C00u, // IMAGE_STORE v[12:15], v[0:2], s[16:23], dim:3D
            0xBF810000u,
        };

        constexpr uint32_t LUT_W = 4, LUT_H = 4, LUT_D = 4;
        constexpr size_t LUT_TEXELS = LUT_W * LUT_H * LUT_D;
        constexpr uint32_t LUT_TILE = static_cast<uint32_t>(TileMode::Sw64KbRX);
        const size_t lut_guest_bytes =
            tiled_volume_bytes(LUT_W, LUT_H, LUT_D, LUT_TILE, sizeof(uint32_t));
        CHECK(lut_guest_bytes != 0 && tile_mode_supports_volume(LUT_TILE),
              "Syberia reduced LUT uses the implemented tiled 3D SW_64KB_R_X layout");

        auto float_bits = [](float value) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        };
        std::vector<uint32_t> source_bits(LUT_TEXELS * 4u);
        std::vector<uint32_t> expected_bits(LUT_TEXELS * 4u);
        std::vector<uint32_t> expected_packed(LUT_TEXELS);
        bool has_sub_one = false, has_hdr = false, has_quantized_value = false;
        for (uint32_t z = 0; z < LUT_D; ++z) {
            for (uint32_t y = 0; y < LUT_H; ++y) {
                for (uint32_t x = 0; x < LUT_W; ++x) {
                    const size_t texel = (static_cast<size_t>(z) * LUT_H + y) * LUT_W + x;
                    const float values[4] = {
                        0.003f + x * 0.537f + y * 0.019f + z * 0.007f,
                        0.125f + x * 0.013f + y * 0.911f + z * 0.203f,
                        0.250f + x * 0.017f + y * 0.071f + z * 2.003f,
                        1.0f,
                    };
                    const float quantized[4] = {
                        f11_to_float(float_to_f11(values[0])),
                        f11_to_float(float_to_f11(values[1])),
                        f10_to_float(float_to_f10(values[2])),
                        1.0f,
                    };
                    for (uint32_t channel = 0; channel < 4; ++channel) {
                        source_bits[texel * 4 + channel] = float_bits(values[channel]);
                        expected_bits[texel * 4 + channel] = float_bits(quantized[channel]);
                        if (channel < 3) {
                            has_sub_one |= values[channel] > 0.0f && values[channel] < 1.0f;
                            has_hdr |= values[channel] > 1.0f;
                            has_quantized_value |= float_bits(values[channel]) !=
                                                   float_bits(quantized[channel]);
                        }
                    }
                    expected_packed[texel] =
                        static_cast<uint32_t>(float_to_f11(values[0])) |
                        (static_cast<uint32_t>(float_to_f11(values[1])) << 11) |
                        (static_cast<uint32_t>(float_to_f10(values[2])) << 22);
                }
            }
        }
        CHECK(has_sub_one && has_hdr && has_quantized_value,
              "Syberia reduced LUT oracle spans sub-one, HDR, and real packing boundaries");

        auto spirv_has_opcode = [](const std::vector<uint32_t>& spirv, uint16_t wanted) {
            for (size_t offset = 5; offset < spirv.size();) {
                const uint32_t word_count = spirv[offset] >> 16;
                if (!word_count || offset + word_count > spirv.size()) return false;
                if (static_cast<uint16_t>(spirv[offset]) == wanted) return true;
                offset += word_count;
            }
            return false;
        };

        auto run_r11_volume_handoff = [&](uint32_t native_support,
                                           bool native_capability_present) {
            const char* arm = native_capability_present ? "capability-present" : "portable";
            std::vector<uint8_t> lut_guest(lut_guest_bytes, 0xA5);
            const uint32_t sentinel_word = static_cast<uint32_t>(float_to_f11(0.0625f)) |
                (static_cast<uint32_t>(float_to_f11(0.125f)) << 11) |
                (static_cast<uint32_t>(float_to_f10(0.25f)) << 22);
            std::vector<uint32_t> sentinel_linear(LUT_TEXELS, sentinel_word);
            const bool sentinel_tiled = tile_volume(
                lut_guest.data(), lut_guest.size(),
                reinterpret_cast<const uint8_t*>(sentinel_linear.data()),
                LUT_W, LUT_H, LUT_D, LUT_TILE, sizeof(uint32_t));
            CHECK(sentinel_tiled,
                  native_capability_present
                      ? "native-capability R11 LUT sentinel tiles across XYZ"
                      : "portable R11 LUT sentinel tiles across XYZ");

            ShaderResource producer_src{};
            producer_src.cls = ResourceClass::StorageImage;
            producer_src.img_dim = 2;
            producer_src.binding = 4;
            producer_src.sgpr_base = 0;
            producer_src.format = DataFormat::Float32;
            producer_src.num_components = 4;
            producer_src.width = LUT_W;
            producer_src.height = LUT_H;
            producer_src.depth = LUT_D;
            producer_src.gpu_addr = reinterpret_cast<uint64_t>(source_bits.data());
            producer_src.size = static_cast<uint32_t>(source_bits.size() * sizeof(uint32_t));

            ShaderResource producer_dst{};
            producer_dst.cls = ResourceClass::StorageImage;
            producer_dst.img_dim = 2;
            producer_dst.binding = 5;
            producer_dst.sgpr_base = 8;
            producer_dst.format = DataFormat::Float10_11_11;
            producer_dst.num_components = 3;
            producer_dst.width = LUT_W;
            producer_dst.height = LUT_H;
            producer_dst.depth = LUT_D;
            producer_dst.tile_mode = LUT_TILE;
            producer_dst.gpu_addr = reinterpret_cast<uint64_t>(lut_guest.data());
            producer_dst.size = static_cast<uint32_t>(lut_guest.size());

            ShaderResourceTable producer_rt;
            producer_rt.resources = {producer_src, producer_dst};
            ComputeShaderConfig producer_config;
            producer_config.user_sgprs.resize(16);
            producer_config.local_x = LUT_W;
            producer_config.local_y = LUT_H;
            producer_config.local_z = LUT_D;
            producer_config.threads_x = LUT_W;
            producer_config.threads_y = LUT_H;
            producer_config.threads_z = LUT_D;
            producer_config.tidig_comp_cnt = 2;
            producer_config.native_storage_format_support = native_support;
            producer_config.packed_r11_storage = true;
            const std::vector<uint32_t> producer_spirv = recompile_compute(
                r11_volume_producer, std::size(r11_volume_producer),
                &producer_rt, producer_config);
            const DescriptorValidationReport producer_report =
                validate_spirv_descriptor_interface(
                    producer_spirv, &producer_rt, 0, SpirvShaderStage::Compute, false);
            const SpirvDescriptorBinding* reflected_lut =
                find_spirv_descriptor_binding(producer_report, 0, producer_dst.binding);
            const bool producer_shape_ok = !producer_spirv.empty() && producer_report.ok() &&
                reflected_lut && reflected_lut->kind == SpirvDescriptorKind::StorageImage &&
                reflected_lut->image_dim == 2 && reflected_lut->writable;
            CHECK(producer_shape_ok,
                  native_capability_present
                      ? "native-capability R11 producer reflects writable 3D storage"
                      : "portable R11 producer reflects writable 3D storage");
            const bool exact_storage = producer_shape_ok && !reflected_lut->storage_float &&
                reflected_lut->storage_image_format == kSpirvImageFormatR32ui;
            CHECK(exact_storage,
                  native_capability_present
                      ? "native R11 capability preserves exact packed R32ui 3D storage"
                      : "disabled native capability selects exact packed R32ui 3D storage");
            if (!exact_storage) return;

            std::vector<uint32_t> output_bits(LUT_TEXELS * 4u, 0xCDCDCDCDu);
            ShaderResource sampled_lut = producer_dst;
            sampled_lut.cls = ResourceClass::Texture;
            sampled_lut.binding = 4;
            sampled_lut.sgpr_base = 0;
            sampled_lut.sampler_sgpr_base = 8;
            sampled_lut.mag_filter = 1;
            sampled_lut.min_filter = 1;
            sampled_lut.mip_filter = 0;

            ShaderResource consumer_dst{};
            consumer_dst.cls = ResourceClass::StorageImage;
            consumer_dst.img_dim = 2;
            consumer_dst.binding = 5;
            consumer_dst.sgpr_base = 16;
            consumer_dst.format = DataFormat::Float32;
            consumer_dst.num_components = 4;
            consumer_dst.width = LUT_W;
            consumer_dst.height = LUT_H;
            consumer_dst.depth = LUT_D;
            consumer_dst.gpu_addr = reinterpret_cast<uint64_t>(output_bits.data());
            consumer_dst.size = static_cast<uint32_t>(output_bits.size() * sizeof(uint32_t));

            ShaderResourceTable consumer_rt;
            consumer_rt.resources = {sampled_lut, consumer_dst};
            ComputeShaderConfig consumer_config;
            consumer_config.user_sgprs.resize(24);
            consumer_config.local_x = LUT_W;
            consumer_config.local_y = LUT_H;
            consumer_config.local_z = LUT_D;
            consumer_config.threads_x = LUT_W;
            consumer_config.threads_y = LUT_H;
            consumer_config.threads_z = LUT_D;
            consumer_config.tidig_comp_cnt = 2;
            consumer_config.native_storage_format_support = native_support;
            const std::vector<uint32_t> consumer_spirv = recompile_compute(
                r11_volume_consumer, std::size(r11_volume_consumer),
                &consumer_rt, consumer_config);
            const DescriptorValidationReport consumer_report =
                validate_spirv_descriptor_interface(
                    consumer_spirv, &consumer_rt, 0, SpirvShaderStage::Compute, false);
            const SpirvDescriptorBinding* reflected_sample =
                find_spirv_descriptor_binding(consumer_report, 0, sampled_lut.binding);
            constexpr uint16_t OpImageSampleExplicitLod = 88;
            const bool consumer_shape_ok = !consumer_spirv.empty() && consumer_report.ok() &&
                reflected_sample &&
                reflected_sample->kind == SpirvDescriptorKind::CombinedImageSampler &&
                reflected_sample->image_dim == 2 && reflected_sample->sampled_float &&
                reflected_sample->normalized_sampling &&
                spirv_has_opcode(consumer_spirv, OpImageSampleExplicitLod);
            CHECK(consumer_shape_ok,
                  native_capability_present
                      ? "native-capability arm retains explicit-LOD normalized 3D R11 sampling"
                      : "portable arm retains explicit-LOD normalized 3D R11 sampling");
            if (!consumer_shape_ok) return;

            ComputeItem producer_item;
            producer_item.spirv = producer_spirv;
            producer_item.resources = std::make_shared<ShaderResourceTable>(producer_rt);
            producer_item.dispatch_index = native_capability_present ? 218 : 118;
            producer_item.command_order = 10;
            producer_item.code_addr = native_capability_present ? 0x1790a118u : 0x1790b118u;
            producer_item.launch.threads_x = LUT_W;
            producer_item.launch.threads_y = LUT_H;
            producer_item.launch.threads_z = LUT_D;
            producer_item.launch.local_x = LUT_W;
            producer_item.launch.local_y = LUT_H;
            producer_item.launch.local_z = LUT_D;
            producer_item.launch.groups_x = producer_item.launch.groups_y =
                producer_item.launch.groups_z = 1;

            ComputeItem consumer_item = producer_item;
            consumer_item.spirv = consumer_spirv;
            consumer_item.resources = std::make_shared<ShaderResourceTable>(consumer_rt);
            consumer_item.dispatch_index = native_capability_present ? 219 : 119;
            consumer_item.command_order = 20;
            consumer_item.code_addr = native_capability_present ? 0x1790a119u : 0x1790b119u;

            // Poison-prove the producer against an independent destination. The ordered handoff
            // below must exercise the steady-state retained image, while the real LUT remains the
            // old sentinel so the separately primed sampled cache is a meaningful stale positive.
            std::vector<uint8_t> proof_guest = lut_guest;
            ShaderResourceTable proof_rt = producer_rt;
            proof_rt.resources.back().gpu_addr =
                reinterpret_cast<uint64_t>(proof_guest.data());
            ComputeItem proof_item = producer_item;
            proof_item.resources = std::make_shared<ShaderResourceTable>(proof_rt);
            proof_item.dispatch_index += 2000;
            proof_item.command_order = 2;
            CHECK(prosper::frontend::execute_live_compute_items({proof_item}),
                  native_capability_present
                      ? "native-capability packed R11 producer proves complete coverage"
                      : "portable packed R11 producer proves complete coverage");

            // Prime the sampled-image cache with the old guest LUT. The ordered run must still see
            // the producer's new result; otherwise a first-use upload would accidentally make the
            // test green without exercising same-submit invalidation/authority.
            std::vector<uint32_t> warm_bits(LUT_TEXELS * 4u, 0xABABABABu);
            ShaderResourceTable warm_rt = consumer_rt;
            warm_rt.resources.back().gpu_addr = reinterpret_cast<uint64_t>(warm_bits.data());
            ComputeItem warm_item = consumer_item;
            warm_item.resources = std::make_shared<ShaderResourceTable>(warm_rt);
            warm_item.dispatch_index += 1000;
            warm_item.command_order = 1;
            const bool warm_executed =
                prosper::frontend::execute_live_compute_items({warm_item});
            const uint32_t warm_expected[4] = {
                float_bits(f11_to_float(static_cast<uint16_t>(sentinel_word))),
                float_bits(f11_to_float(static_cast<uint16_t>(sentinel_word >> 11))),
                float_bits(f10_to_float(static_cast<uint16_t>(sentinel_word >> 22))),
                float_bits(1.0f),
            };
            size_t warm_mismatches = 0;
            for (size_t texel = 0; texel < LUT_TEXELS; ++texel)
                for (uint32_t channel = 0; channel < 4; ++channel)
                    warm_mismatches += warm_bits[texel * 4 + channel] !=
                                       warm_expected[channel];
            CHECK(warm_executed && warm_mismatches == 0,
                  native_capability_present
                      ? "native-capability sampled cache primes from the old tiled R11 sentinel"
                      : "portable sampled cache primes from the old tiled R11 sentinel");

            std::array<uint32_t, 2> callback_dispatches = {UINT32_MAX, UINT32_MAX};
            size_t callback_count = 0;
            bool callbacks_ok = true;
            bool packed_graphics_imported = false;
            const uint64_t packed_transfer_seeds_before =
                prosper::frontend::live_compute_storage_transfer_seeds();
            const OrderedSubmitResult submit = execute_ordered_items(
                {{SubmitOperationKind::Dispatch, producer_item.dispatch_index,
                  producer_item.command_order},
                 {SubmitOperationKind::Dispatch, consumer_item.dispatch_index,
                  consumer_item.command_order}},
                {}, {producer_item, consumer_item},
                [](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                    return RenderedFrame{};
                },
                [&](const std::vector<ComputeItem>& items) {
                    const bool singleton = items.size() == 1;
                    if (callback_count < callback_dispatches.size() && singleton)
                        callback_dispatches[callback_count] = items[0].dispatch_index;
                    ++callback_count;
                     const bool ok = singleton &&
                         prosper::frontend::execute_live_compute_items(items);
                     if (ok && items[0].dispatch_index == producer_item.dispatch_index) {
                         prosper::frontend::LiveComputeImageImport graphics_import;
                         packed_graphics_imported =
                             prosper::frontend::import_live_compute_storage_image(
                                 sampled_lut, lut_guest.size(), graphics_import) &&
                             graphics_import.valid() && graphics_import.native_format == 122u;
                     }
                     callbacks_ok &= ok;
                    return ok;
                },
                1, 1);
            const uint64_t packed_transfer_seeds_after =
                prosper::frontend::live_compute_storage_transfer_seeds();
            CHECK(submit.compute_executed && callbacks_ok && callback_count == 2,
                  native_capability_present
                      ? "native-capability ordered submit executes both dispatches exactly once"
                      : "portable ordered submit executes both dispatches exactly once");
            CHECK(callback_dispatches[0] == producer_item.dispatch_index &&
                      callback_dispatches[1] == consumer_item.dispatch_index,
                  native_capability_present
                      ? "native-capability submit preserves producer-before-consumer order"
                      : "portable submit preserves producer-before-consumer order");
            CHECK(!native_2d_compute_transfer_available ||
                      packed_transfer_seeds_after > packed_transfer_seeds_before,
                  native_capability_present
                      ? "native-capability packed R11 producer seeds sampled volume on-GPU"
                      : "portable packed R11 producer seeds sampled volume on-GPU");
            CHECK(native_2d_compute_transfer_available ||
                      packed_transfer_seeds_after == packed_transfer_seeds_before,
                  native_capability_present
                      ? "disabled transfer keeps native-capability packed R11 on the guest fallback"
                      : "disabled transfer keeps portable packed R11 on the guest fallback");
            CHECK(!native_2d_compute_transfer_available || packed_graphics_imported,
                  native_capability_present
                      ? "native-capability packed R11 storage result exports to graphics on-GPU"
                      : "portable packed R11 storage result exports to graphics on-GPU");

            std::vector<uint32_t> produced_packed(LUT_TEXELS, 0xDEADBEEFu);
            const bool produced_detiled = detile_volume(
                reinterpret_cast<uint8_t*>(produced_packed.data()), lut_guest.data(),
                lut_guest.size(), LUT_W, LUT_H, LUT_D, LUT_TILE, sizeof(uint32_t));
            CHECK(produced_detiled,
                  native_capability_present
                      ? "native-capability producer detiles for direct packed-word census"
                      : "portable producer detiles for direct packed-word census");
            const uint32_t field_shift[3] = {0, 11, 22};
            const uint32_t field_mask[3] = {0x7ffu, 0x7ffu, 0x3ffu};
            std::array<size_t, 3> packed_channel_mismatches{};
            size_t packed_mismatches = 0, packed_lower = 0, packed_higher = 0;
            for (size_t texel = 0; produced_detiled && texel < LUT_TEXELS; ++texel) {
                if (produced_packed[texel] == expected_packed[texel]) continue;
                ++packed_mismatches;
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    const uint32_t actual =
                        (produced_packed[texel] >> field_shift[channel]) & field_mask[channel];
                    const uint32_t expected =
                        (expected_packed[texel] >> field_shift[channel]) & field_mask[channel];
                    if (actual == expected) continue;
                    ++packed_channel_mismatches[channel];
                    packed_lower += actual < expected;
                    packed_higher += actual > expected;
                }
                if (packed_mismatches <= 8) {
                    const size_t xy = LUT_W * LUT_H;
                    const uint32_t z = static_cast<uint32_t>(texel / xy);
                    const uint32_t y = static_cast<uint32_t>((texel % xy) / LUT_W);
                    const uint32_t x = static_cast<uint32_t>(texel % LUT_W);
                    std::printf("  %s R11 producer packed diff xyz=(%u,%u,%u) expected=%08x "
                                "actual=%08x xor=%08x\n", arm, x, y, z,
                                expected_packed[texel], produced_packed[texel],
                                expected_packed[texel] ^ produced_packed[texel]);
                }
            }

            size_t sentinel_survivors = 0, mismatches = 0;
            std::array<size_t, 4> output_channel_mismatches{};
            size_t first_texel = SIZE_MAX;
            uint32_t first_channel = UINT32_MAX;
            for (size_t texel = 0; texel < LUT_TEXELS; ++texel) {
                for (uint32_t channel = 0; channel < 4; ++channel) {
                    const size_t index = texel * 4 + channel;
                    sentinel_survivors += output_bits[index] == 0xCDCDCDCDu;
                    if (output_bits[index] == expected_bits[index]) continue;
                    if (first_texel == SIZE_MAX) {
                        first_texel = texel;
                        first_channel = channel;
                    }
                    ++output_channel_mismatches[channel];
                    ++mismatches;
                }
            }
            if (packed_mismatches || mismatches) {
                std::printf("  %s R11 producer census: packed-texels=%zu/%zu channels=(%zu,%zu,%zu) "
                            "lower=%zu higher=%zu\n", arm, packed_mismatches, LUT_TEXELS,
                            packed_channel_mismatches[0], packed_channel_mismatches[1],
                            packed_channel_mismatches[2], packed_lower, packed_higher);
            }
            if (mismatches) {
                std::printf("  %s tiled R11 3D census: mismatches=%zu/%zu first-texel=%zu "
                            "channel=%u channels=(%zu,%zu,%zu,%zu) expected=%08x actual=%08x "
                            "sentinel-survivors=%zu\n",
                            arm, mismatches, expected_bits.size(), first_texel, first_channel,
                            output_channel_mismatches[0], output_channel_mismatches[1],
                            output_channel_mismatches[2], output_channel_mismatches[3],
                            expected_bits[first_texel * 4 + first_channel],
                            output_bits[first_texel * 4 + first_channel], sentinel_survivors);
            }
            CHECK(sentinel_survivors == 0,
                  native_capability_present
                      ? "native-capability 3D consumer overwrites every output channel"
                      : "portable 3D consumer overwrites every output channel");
            CHECK(packed_mismatches == 0,
                  native_capability_present
                      ? "native-capability producer matches every packed R11 voxel"
                      : "portable producer matches every packed R11 voxel");
            CHECK(mismatches == 0,
                  native_capability_present
                      ? "native-capability exact R11 handoff matches every CPU f11/f10 voxel"
                      : "portable exact R11 handoff matches every CPU f11/f10 voxel");
        };

        run_r11_volume_handoff(0, false);
        run_r11_volume_handoff(
            native_storage_3d_format_support_bit(DataFormat::Float10_11_11, 3),
            true);
    }

    // (b) PARTIAL store under a FULL grid (reviewer B1): a write-only 2D kernel that always stores to
    // row 0 (v5 hard-zero), dispatched over a W x 2 grid so covers_extent is TRUE while row 1 is never
    // written. The proof must detect the survivor, NOT fast-skip, and preserve row 1's prior content.
    // Before the fix this row was undefined pool memory packed to the guest -- silent corruption.
    {
        static const uint32_t store_row0_2d[] = {
            0x7E080300u,             // v4 = v0 (x)
            0x7E0A0280u,             // v5 = 0  (y) -- ALWAYS row 0, regardless of the y invocation
            0xF0200F08u, 0x00020004u,// IMAGE_STORE v0..v3 at (v4,v5) to binding 5 -- write-only
            0xBF810000u,             // s_endpgm
        };
        const uint32_t H2 = 2;
        std::vector<uint32_t> p_index(W);
        for (uint32_t i = 0; i < W; ++i) p_index[i] = i;   // x == gid: row 0 fully written, row 1 never
        std::vector<uint32_t> pdummy(4, 0);
        std::vector<uint8_t> part_guest(W * H2 * 4);
        for (size_t i = 0; i < part_guest.size(); ++i) part_guest[i] = (uint8_t)(i * 29 + 13);
        const std::vector<uint8_t> part_original = part_guest;
        ShaderResourceTable part_rt;
        auto padd_buf = [&](uint32_t b, void* d, uint32_t s) {
            ShaderResource r{}; r.cls = ResourceClass::ConstantBuffer; r.binding = b;
            r.gpu_addr = (uint64_t)(uintptr_t)d; r.size = s; part_rt.resources.push_back(r); };
        padd_buf(0, p_index.data(), W * sizeof(uint32_t));
        padd_buf(1, pdummy.data(), 16); padd_buf(2, pdummy.data(), 16); padd_buf(3, pdummy.data(), 16);
        ShaderResource pdst{};
        pdst.cls = ResourceClass::StorageImage; pdst.img_dim = 1; pdst.binding = 5; pdst.sgpr_base = 8;
        pdst.format = DataFormat::Unorm8; pdst.num_components = 4; pdst.width = W; pdst.height = H2;
        pdst.depth = 1; pdst.gpu_addr = (uint64_t)(uintptr_t)part_guest.data(); pdst.size = W * H2 * 4;
        part_rt.resources.push_back(pdst);
        std::vector<uint32_t> part_spirv = recompile_valu(
            store_row0_2d, sizeof(store_row0_2d) / sizeof(store_row0_2d[0]), 1, 0, &part_rt);
        CHECK(!part_spirv.empty(), "write-only 2D row-0 store kernel recompiles");
        if (!part_spirv.empty()) {
            ComputeItem it; it.spirv = part_spirv;
            it.resources = std::make_shared<ShaderResourceTable>(part_rt);
            it.launch.threads_x = W; it.launch.local_x = 64; it.launch.groups_x = 1;
            it.launch.threads_y = H2; it.launch.local_y = 1; it.launch.groups_y = H2;
            it.launch.threads_z = 1; it.launch.local_z = 1; it.launch.groups_z = 1;
            it.code_addr = 0x1122f22du;
            const size_t row_bytes = (size_t)W * 4;
            // Two runs: the first proves (poison) and detects partial coverage; the second uses the
            // cached "Partial" verdict and seeds normally. Row 1 must survive on BOTH.
            for (int run = 0; run < 2; ++run) {
                CHECK(prosper::frontend::execute_live_compute_items({it}),
                      run == 0 ? "partial-store proving run executes"
                               : "partial-store cached (always-seed) run executes");
                const bool row1_preserved = std::memcmp(part_guest.data() + row_bytes,
                                                        part_original.data() + row_bytes, row_bytes) == 0;
                CHECK(row1_preserved,
                      run == 0 ? "B1: full-grid partial store preserves untouched row on the proving run"
                               : "B1: full-grid partial store preserves untouched row once proven partial");
                const bool row0_written = std::memcmp(part_guest.data(),
                                                      part_original.data(), row_bytes) != 0;
                CHECK(row0_written, "partial store did write the covered row (kernel ran)");
            }
        }
    }

    // #3285: untouched write-only storage image test. A compute dispatch binds a write-only
    // storage image, but the shader writes 0 texels into the image bounds. 100% of the proving
    // poison survives. Verify it is classified as SeedCoverage::None, writeback is skipped on the
    // proving run (preserving guest memory without spurious host write notifications), and future
    // dispatches fast-skip both seeding and writeback entirely.
    {
        static const uint32_t store_untouched_2d[] = {
            0x7E0802FFu, 0x0000270Fu,// v4 = 9999 (x, well out-of-bounds)
            0x7E0A02FFu, 0x0000270Fu,// v5 = 9999 (y, well out-of-bounds)
            0xF0200F08u, 0x00020004u,// IMAGE_STORE v0..v3 at (v4,v5) to binding 5 -- write-only
            0xBF810000u,             // s_endpgm
        };
        const uint32_t H2 = 2;
        std::vector<uint32_t> p_index(W);
        for (uint32_t i = 0; i < W; ++i) p_index[i] = i;
        std::vector<uint32_t> pdummy(4, 0);
        std::vector<uint8_t> untouched_guest(W * H2 * 4);
        for (size_t i = 0; i < untouched_guest.size(); ++i)
            untouched_guest[i] = (uint8_t)(i * 31 + 17);
        const std::vector<uint8_t> untouched_original = untouched_guest;
        ShaderResourceTable untouched_rt;
        auto add_buf = [&](uint32_t b, void* d, uint32_t s) {
            ShaderResource r{}; r.cls = ResourceClass::ConstantBuffer; r.binding = b;
            r.gpu_addr = (uint64_t)(uintptr_t)d; r.size = s; untouched_rt.resources.push_back(r); };
        add_buf(0, p_index.data(), W * sizeof(uint32_t));
        add_buf(1, pdummy.data(), 16); add_buf(2, pdummy.data(), 16); add_buf(3, pdummy.data(), 16);
        ShaderResource udst{};
        udst.cls = ResourceClass::StorageImage; udst.img_dim = 1; udst.binding = 5; udst.sgpr_base = 8;
        udst.format = DataFormat::Unorm8; udst.num_components = 4; udst.width = W; udst.height = H2;
        udst.depth = 1; udst.gpu_addr = (uint64_t)(uintptr_t)untouched_guest.data(); udst.size = W * H2 * 4;
        untouched_rt.resources.push_back(udst);
        std::vector<uint32_t> untouched_spirv = recompile_valu(
            store_untouched_2d, sizeof(store_untouched_2d) / sizeof(store_untouched_2d[0]), 1, 0, &untouched_rt);
        CHECK(!untouched_spirv.empty(), "write-only 2D out-of-bounds store kernel recompiles");
        if (!untouched_spirv.empty()) {
            ComputeItem it; it.spirv = untouched_spirv;
            it.resources = std::make_shared<ShaderResourceTable>(untouched_rt);
            it.launch.threads_x = W; it.launch.local_x = 64; it.launch.groups_x = 1;
            it.launch.threads_y = H2; it.launch.local_y = 1; it.launch.groups_y = H2;
            it.launch.threads_z = 1; it.launch.local_z = 1; it.launch.groups_z = 1;
            it.code_addr = 0x32850001u;
            // Run 0: proves (100% poison survived -> SeedCoverage::None), skips writeback.
            // Run 1: fast-skip cached None verdict, skips seeding and writeback.
            for (int run = 0; run < 2; ++run) {
                CHECK(prosper::frontend::execute_live_compute_items({it}),
                      run == 0 ? "untouched-store proving run executes"
                               : "untouched-store fast-skip run executes");
                const bool preserved = std::memcmp(untouched_guest.data(),
                                                   untouched_original.data(),
                                                   untouched_original.size()) == 0;
                CHECK(preserved,
                      run == 0 ? "untouched-store proving run leaves guest memory unmodified"
                               : "untouched-store fast-skip run leaves guest memory unmodified");
            }
        }
    }

    // Keep this last: the production contract deliberately latches a lost VkDevice for the rest of
    // the process. Inject one queue-submit loss into a two-item batch, then call the backend again.
    // Both checks measure the lever directly: disabling either the in-batch break or the persistent
    // short-circuit causes another queue-submit attempt and makes a named assertion fail.
    {
        FILE* diagnostic_file = std::tmpfile();
        int saved_stderr = -1;
        bool capturing = false;
        if (diagnostic_file) {
            std::fflush(stderr);
            saved_stderr = duplicate_descriptor(file_descriptor(stderr));
            capturing = saved_stderr >= 0 &&
                        replace_descriptor(file_descriptor(diagnostic_file),
                                           file_descriptor(stderr)) >= 0;
        }
        CHECK(capturing, "device-loss diagnostic capture initialized");

        const uint64_t attempts_before =
            prosper::frontend::live_compute_queue_submit_attempts();
        prosper::frontend::live_compute_force_next_queue_submit_device_lost_for_test();
        const bool batch_result =
            prosper::frontend::execute_live_compute_items({item, item});
        const uint64_t attempts_after_loss =
            prosper::frontend::live_compute_queue_submit_attempts();
        const bool later_result = prosper::frontend::execute_live_compute_items({item});
        const uint64_t attempts_after_later_call =
            prosper::frontend::live_compute_queue_submit_attempts();

        std::string diagnostic;
        if (capturing) {
            std::fflush(stderr);
            if (replace_descriptor(saved_stderr, file_descriptor(stderr)) < 0)
                capturing = false;
            close_descriptor(saved_stderr);
            saved_stderr = -1;
            std::rewind(diagnostic_file);
            char buffer[512];
            while (const size_t bytes = std::fread(buffer, 1, sizeof(buffer), diagnostic_file))
                diagnostic.append(buffer, bytes);
        } else if (saved_stderr >= 0) {
            close_descriptor(saved_stderr);
        }
        if (diagnostic_file) std::fclose(diagnostic_file);

        CHECK(!batch_result && attempts_after_loss == attempts_before + 1,
              "device loss aborts the current batch before another queue submit");
        CHECK(!later_result && attempts_after_later_call == attempts_after_loss,
              "latched device loss rejects later callbacks before queue submission");
        CHECK(capturing &&
                  diagnostic.find("result=VK_ERROR_DEVICE_LOST(-4)") != std::string::npos &&
                  diagnostic.find("program=0x401aec200 submit=11 dispatch=7 order=70") !=
                      std::string::npos &&
                  diagnostic.find("result=VK_ERROR_DEVICE_LOST(-4)",
                                  diagnostic.find("result=VK_ERROR_DEVICE_LOST(-4)") + 1) ==
                      std::string::npos,
              "device loss is unconditional and identifies the first observed guest dispatch");
    }

    // #3155: report the write-watch census this run produced, and pin the structural identities
    // that make it readable. The numbers themselves are workload-dependent and are deliberately NOT
    // asserted -- this suite is not The Plucky Squire -- but a census whose buckets do not partition
    // its own denominator is an instrument nobody should quote, and that IS checkable here.
    {
        const auto census = prosper::frontend::live_compute_write_watch_census();
        char census_line[1024];
        const size_t census_used = prosper::frontend::format_write_watch_census(
            census, census_line, sizeof census_line);
        if (census_used) std::fwrite(census_line, 1, census_used, stdout);
        CHECK(census.decisions > 0,
              "the suite consults the write-watch promotion policy at least once");
        CHECK(census.stability_0 + census.stability_1 + census.stability_2 +
                  census.stability_3_plus == census.decisions,
              "census stability buckets partition the decisions they are a fraction of");
        CHECK(census.granted <= census.threshold_met &&
                  census.budget_refused == census.threshold_met - census.granted,
              "a granted promotion is a subset of the threshold-eligible ones");
        CHECK(census.exact_compares > 0 && census.exact_compare_bytes > 0,
              "the suite reaches the full byte comparison the census exists to measure");
    }

    if (fails) {
        std::printf("== FAIL: %d == (%d assertions executed)\n", fails, checks);
        return 1;
    }
    std::printf("== PASS == (%d assertions executed)\n", checks);
    return 0;
}
