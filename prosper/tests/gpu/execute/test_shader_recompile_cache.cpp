#include "gpu/execute/gpu_execute.hpp"
#include "gpu/diagnostics/shader_dump_filter.hpp"
#include "hle/dispatch/dispatch.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "fixtures/gta5_nullable_output_fixture.hpp"
#include "fixtures/test_scratch.h"

using namespace prosper::gpu;

static int failures = 0;
#define CHECK(condition, message) do { \
    if (condition) std::printf("  [ok]   %s\n", message); \
    else { std::printf("  [FAIL] %s\n", message); ++failures; } \
} while (0)

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

// Names, not just extensions: #3196 is about what the FILENAME says, so the assertion has to read
// it. A substring match is deliberate -- pinning the whole name would pin the hashes too, and those
// legitimately change whenever the recompiler's output does.
static size_t count_names_containing(const std::filesystem::path& directory, const char* needle) {
    std::error_code ec;
    size_t count = 0;
    for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (it->is_regular_file(ec) &&
            it->path().filename().string().find(needle) != std::string::npos)
            ++count;
    }
    return ec ? 0 : count;
}

static size_t count_extension(const std::filesystem::path& directory, const char* extension) {
    std::error_code ec;
    size_t count = 0;
    for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (it->is_regular_file(ec) && it->path().extension() == extension) ++count;
    }
    return ec ? 0 : count;
}

struct CapturedStderr {
    std::filesystem::path path;
    std::string output;
};

struct ScratchCaptureFile {
    std::filesystem::path path;
    FILE* stream = nullptr;

    explicit ScratchCaptureFile(std::filesystem::path capture_path)
        : path(std::move(capture_path)) {
#ifdef _WIN32
        stream = _wfopen(path.c_str(), L"w+b");
#else
        stream = std::fopen(path.c_str(), "w+b");
#endif
    }

    ~ScratchCaptureFile() {
        if (stream) std::fclose(stream);
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    ScratchCaptureFile(const ScratchCaptureFile&) = delete;
    ScratchCaptureFile& operator=(const ScratchCaptureFile&) = delete;
};

static std::filesystem::path next_stderr_capture_path() {
    static std::atomic<uint64_t> capture_sequence{0};
    return prosper_test::test_scratch_dir() /
        ("shader-recompile-stderr-" +
         std::to_string(capture_sequence.fetch_add(1, std::memory_order_relaxed)) + ".log");
}

static size_t count_occurrences(const std::string& text, const std::string& needle) {
    size_t count = 0;
    for (size_t offset = 0; (offset = text.find(needle, offset)) != std::string::npos;
         offset += needle.size())
        ++count;
    return count;
}

template <typename F>
static CapturedStderr capture_stderr(F&& body) {
    CapturedStderr result;
    result.path = next_stderr_capture_path();
    ScratchCaptureFile capture(result.path);
    if (!capture.stream || std::fflush(stderr) != 0) return result;
#ifdef _WIN32
    const int stderr_fd = _fileno(stderr);
    const int saved = _dup(stderr_fd);
    const bool redirected = saved >= 0 && _dup2(_fileno(capture.stream), stderr_fd) == 0;
#else
    const int stderr_fd = fileno(stderr);
    const int saved = dup(stderr_fd);
    const bool redirected = saved >= 0 && dup2(fileno(capture.stream), stderr_fd) >= 0;
#endif
    if (!redirected) {
        if (saved >= 0) {
#ifdef _WIN32
            _close(saved);
#else
            close(saved);
#endif
        }
        return result;
    }

    body();
    std::fflush(stderr);
#ifdef _WIN32
    const bool restored = _dup2(saved, stderr_fd) == 0;
    _close(saved);
#else
    const bool restored = dup2(saved, stderr_fd) >= 0;
    close(saved);
#endif
    if (!restored) return result;
    std::rewind(capture.stream);
    char bytes[4096];
    for (size_t count; (count = std::fread(bytes, 1, sizeof bytes, capture.stream)) != 0;)
        result.output.append(bytes, count);
    return result;
}

// Fullscreen triangle and solid green shaders assembled for gfx1030. These are also used by the
// end-to-end GPU executor tests, but this test needs no Vulkan device.
static const uint32_t kVs[] = {
    0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u,
    0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u, 0x08040D02u,
    0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
};
static const uint32_t kPs[] = {
    0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,
    0xF800180Fu, 0x03020100u, 0xBF810000u,
};

// setenv/unsetenv are POSIX and absent on MinGW, where the CI Windows job builds this file.
// `live_renderer.cpp` splits the same way; a null value clears the variable on both platforms.
static void set_env_for_test(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else       unsetenv(name);
#endif
}

int main() {
    std::printf("== test_shader_recompile_cache ==\n");
    clear_shader_recompile_cache();
    clear_shader_analysis_cache();

    ShaderResource resource;
    resource.cls = ResourceClass::VertexBuffer;
    resource.format = DataFormat::Float32;
    resource.num_components = 3;
    resource.binding = 7;
    resource.stride = 12;
    resource.srt_offset = 0x20;
    resource.sgpr_base = 8;
    resource.fetch_pc = 4;
    resource.gpu_addr = 0x100000;
    resource.size = 4096;
    ShaderResourceTable table;
    table.resources.push_back(resource);

    const auto direct_vs = recompile_vertex(kVs, std::size(kVs), &table);
    const std::filesystem::path dump_directory =
        prosper_test::test_scratch_dir() / "prosper-shader-dump-test";
    std::error_code dump_ec;
    std::filesystem::remove_all(dump_directory, dump_ec);
    set_test_env("PROSPER_SHADER_DUMP_SUCCESS", dump_directory.string().c_str());
    uint64_t first_identity = 0;
    const auto cached_vs = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &first_identity);
    CHECK(!direct_vs.empty() && cached_vs == direct_vs,
          "cache miss is byte-identical to the direct vertex recompiler");
    auto stats = shader_recompile_cache_stats();
    CHECK(stats.misses == 1 && stats.hits == 0 && stats.entries == 1,
          "first shader realization records one cache miss");
    CHECK(count_extension(dump_directory, ".bin") == 1 &&
              count_extension(dump_directory, ".spv") == 1,
          "successful shader diagnostics create one raw/SPIR-V pair");

    uint64_t repeated_identity = 0;
    const auto repeated_vs = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &repeated_identity);
    stats = shader_recompile_cache_stats();
    CHECK(repeated_vs == direct_vs && stats.hits == 1 && stats.misses == 1,
          "identical code and descriptor semantics hit the cache");
    CHECK(first_identity != 0 && repeated_identity == first_identity,
          "shader cache hits preserve a non-zero compiled-shader identity");
    auto analysis_stats = shader_analysis_cache_stats();
    CHECK(analysis_stats.misses == 1 && analysis_stats.hits == 1 &&
              analysis_stats.entries == 1,
          "repeated recompilation reuses byte-validated immutable shader analysis");
    CHECK(count_extension(dump_directory, ".bin") == 1 &&
              count_extension(dump_directory, ".spv") == 1,
          "successful shader diagnostics deduplicate cache hits");
    set_test_env("PROSPER_SHADER_DUMP_SUCCESS", nullptr);
    std::filesystem::remove_all(dump_directory, dump_ec);

    // These fields are consumed by the runtime backend, not by the recompiler. Changing them must
    // reuse SPIR-V while each DrawItem continues to carry the new table to descriptor upload.
    table.resources[0].gpu_addr = 0x900000;
    table.resources[0].size = 8192;
    table.resources[0].width = 1024;
    table.resources[0].height = 512;
    table.resources[0].mag_filter = 0;
    table.resources[0].host_data = reinterpret_cast<uint8_t*>(0x1234);
    table.resources[0].host_data_size = 16;
    const auto runtime_changed = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table);
    stats = shader_recompile_cache_stats();
    CHECK(runtime_changed == direct_vs && stats.hits == 2 && stats.misses == 1,
          "runtime-only resource changes preserve the compiled shader cache entry");

    // Binding participates in both descriptor declarations and memory-op lowering, so it must miss.
    table.resources[0].binding = 9;
    const auto direct_rebound = recompile_vertex(kVs, std::size(kVs), &table);
    const auto cached_rebound = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table);
    stats = shader_recompile_cache_stats();
    CHECK(cached_rebound == direct_rebound && stats.misses == 2 && stats.entries == 2,
          "compile-time resource changes miss and remain byte-identical to the oracle");

    // Guest programs with identical bytes may be mapped at different addresses. Shader analysis is
    // address-local, but the compiled shader cache must still compare its immutable code by value.
    std::vector<uint32_t> relocated_vs(kVs, kVs + std::size(kVs));
    const auto relocated_cached = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, relocated_vs.data(), relocated_vs.size(), &table);
    stats = shader_recompile_cache_stats();
    CHECK(relocated_cached == cached_rebound && stats.hits == 3 && stats.misses == 2 &&
              stats.entries == 2,
          "byte-identical shader code at a different address reuses the compiled cache entry");

    // #3130, the mirror of the arm above. With the trip-bound selector ARMED, the same bytes at two
    // addresses must NOT share an entry: the selector targets one program, and the cache is keyed on
    // code bytes, so a collision would hand the bounded module to an excluded program or the
    // unbounded one to the target — silently destroying the isolation that makes a bounded run
    // evidence about ONE program. `rdna2_cfg_support.hpp` states that contract and compute honours
    // it; graphics did not, and could not be caught while fragment diagnostics were pinned at
    // address 0, because a selector cannot match a program with no address.
    {
        std::vector<uint32_t> bound_a(kPs, kPs + std::size(kPs));
        std::vector<uint32_t> bound_b(kPs, kPs + std::size(kPs));
        char selector[32];
        std::snprintf(selector, sizeof selector, "0x%llx",
                      static_cast<unsigned long long>(
                          reinterpret_cast<uintptr_t>(bound_a.data())));
        // MinGW has no setenv/unsetenv; live_renderer.cpp uses the same _putenv_s split.
        set_env_for_test("PROSPER_CFG_TRIP_BOUND", "4");
        set_env_for_test("PROSPER_CFG_TRIP_BOUND_PROGRAM", selector);
        set_env_for_test("PROSPER_CFG_TRIP_BOUND_PHASE", "0");
        const auto before = shader_recompile_cache_stats();
        (void)recompile_graphics_shader_cached(
            ShaderProgramStage::Fragment, bound_a.data(), bound_a.size(), nullptr);
        (void)recompile_graphics_shader_cached(
            ShaderProgramStage::Fragment, bound_b.data(), bound_b.size(), nullptr);
        const auto after = shader_recompile_cache_stats();
        set_env_for_test("PROSPER_CFG_TRIP_BOUND", nullptr);
        set_env_for_test("PROSPER_CFG_TRIP_BOUND_PROGRAM", nullptr);
        set_env_for_test("PROSPER_CFG_TRIP_BOUND_PHASE", nullptr);
        CHECK(after.misses == before.misses + 2 && after.hits == before.hits,
              "#3130: an armed trip-bound selector separates identical bytes at two addresses");
    }

    const auto direct_ps = recompile_fragment(kPs, std::size(kPs), nullptr);
    const auto cached_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr);
    CHECK(!direct_ps.empty() && cached_ps == direct_ps,
          "fragment cache output is byte-identical to the direct recompiler");

    // #3130 threads the guest program address to the fragment recompiler as DIAGNOSTIC provenance.
    // It must NOT reach the cache key: identical bytes produce identical SPIR-V at every guest
    // address, so a relocated copy has to HIT. Without this arm a plumb-through that folded the
    // address into ShaderCompileKey would compile, pass every other case here, and silently give
    // each address its own entry -- a cache defeat visible only as a frame-time regression.
    const auto before_relocated_ps = shader_recompile_cache_stats();
    std::vector<uint32_t> relocated_ps(kPs, kPs + std::size(kPs));
    const auto relocated_cached_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, relocated_ps.data(), relocated_ps.size(), nullptr);
    const auto after_relocated_ps = shader_recompile_cache_stats();
    CHECK(relocated_cached_ps == cached_ps &&
              after_relocated_ps.misses == before_relocated_ps.misses &&
              after_relocated_ps.hits == before_relocated_ps.hits + 1,
          "a fragment program at a different address reuses its compiled cache entry (#3130)");


    // Unreal fragment shaders place small constant tables after S_ENDPGM and address them through an
    // s_getpc_b64-built V#. The owning cache copy must include that proven tail; copying only the walked
    // instruction span makes the cached recompiler reject s_getpc_b64 even though the direct path works.
    std::vector<uint32_t> pcrel_ps = {
        0xbe841f00u,               // s_getpc_b64 s[4:5]
        0x800404b0u,               // s_add_u32 s4, 48, s4 (table begins at byte 52)
        0x82050580u,               // s_addc_u32 s5, 0, s5
        0xbe860390u,               // s_mov_b32 s6, 16 bytes
        0xbe8703ffu, 0x10005004u,  // s_mov_b32 s7, V# format/stride
        0x7e020280u,               // v_mov_b32 v1, 0 (table byte offset)
        0xe0301000u, 0x80010101u,  // buffer_load_dword v1, v1, s[4:7], 0 offen
        0xbf8c3f70u,               // s_waitcnt vmcnt(0)
        0xf800180fu, 0x01010101u,  // exp mrt0 v1,v1,v1,v1
        0xbf810000u,               // s_endpgm
        7u, 11u, 13u, 17u,        // embedded table
    };
    CHECK(rdna2_recompile_code_span(pcrel_ps.data(), pcrel_ps.size()) == pcrel_ps.size(),
          "PC-relative cache span includes the embedded table tail");
    ShaderResourceTable pcrel_table;
    const auto direct_pcrel_ps = recompile_fragment(
        pcrel_ps.data(), pcrel_ps.size(), &pcrel_table);
    const auto cached_pcrel_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, pcrel_ps.data(), pcrel_ps.size(), &pcrel_table);
    CHECK(!direct_pcrel_ps.empty() && cached_pcrel_ps == direct_pcrel_ps,
          "fragment cache retains a proven post-ENDPGM PC-relative table");

    const auto pcrel_stats = shader_recompile_cache_stats();
    pcrel_ps.back() = 19u;
    const auto changed_pcrel_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, pcrel_ps.data(), pcrel_ps.size(), &pcrel_table);
    stats = shader_recompile_cache_stats();
    CHECK(!changed_pcrel_ps.empty() && changed_pcrel_ps != cached_pcrel_ps &&
              stats.misses == pcrel_stats.misses + 1,
          "embedded table contents participate in the shader cache key");

    // S# FORCE_UNNORMALIZED bakes reciprocal spatial extents into sampled-image instructions. The
    // cache must separate normalized and texel-space modules, and two texel-space resources with
    // different extents, while the diagnostic disable arm must reuse the normalized module.
    const auto unnormalized_cache_stats = shader_recompile_cache_stats();
    static const uint32_t kSamplePs[] = {
        0x7e0002ffu, 0x3fc00000u, 0x7e0202ffu, 0x3f000000u,
        0xf0800f08u, 0x00820000u, 0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    ShaderResource sampled_texture;
    sampled_texture.cls = ResourceClass::Texture;
    sampled_texture.format = DataFormat::Unorm8;
    sampled_texture.num_components = 4;
    sampled_texture.binding = 4;
    sampled_texture.sgpr_base = 8;
    sampled_texture.img_dim = 1;
    sampled_texture.width = sampled_texture.height = 2;
    sampled_texture.depth = 1;
    ShaderResourceTable sampled_table;
    sampled_table.resources.push_back(sampled_texture);
    uint64_t normalized_identity = 0, texel2_identity = 0, texel4_identity = 0,
             texel2_repeat_identity = 0, disabled_sample_identity = 0;
    const auto normalized_sample = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kSamplePs, std::size(kSamplePs), &sampled_table,
        nullptr, nullptr, &normalized_identity);
    sampled_table.resources[0].unnormalized = 1;
    const auto texel2_sample = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kSamplePs, std::size(kSamplePs), &sampled_table,
        nullptr, nullptr, &texel2_identity);
    sampled_table.resources[0].width = 4;
    const auto texel4_sample = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kSamplePs, std::size(kSamplePs), &sampled_table,
        nullptr, nullptr, &texel4_identity);
    sampled_table.resources[0].width = 2;
    const auto texel2_repeat = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kSamplePs, std::size(kSamplePs), &sampled_table,
        nullptr, nullptr, &texel2_repeat_identity);
    set_test_env("PROSPER_NO_UNNORMALIZED_COORD_NORMALIZE", "1");
    const auto disabled_sample = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kSamplePs, std::size(kSamplePs), &sampled_table,
        nullptr, nullptr, &disabled_sample_identity);
    set_test_env("PROSPER_NO_UNNORMALIZED_COORD_NORMALIZE", nullptr);
    stats = shader_recompile_cache_stats();
    CHECK(!normalized_sample.empty() && !texel2_sample.empty() && !texel4_sample.empty() &&
              normalized_sample != texel2_sample && texel2_sample != texel4_sample &&
              texel2_repeat == texel2_sample && disabled_sample == normalized_sample &&
              normalized_identity != texel2_identity && texel2_identity != texel4_identity &&
              texel2_repeat_identity == texel2_identity &&
              disabled_sample_identity == normalized_identity &&
              stats.misses == unnormalized_cache_stats.misses + 3 &&
              stats.hits == unnormalized_cache_stats.hits + 2 &&
              stats.entries == unnormalized_cache_stats.entries + 3,
          "shader cache separates normalized and extent-specific texel-coordinate lowering");

    // Astro Bot's title PS uses SMEM rather than MUBUF to consume its getpc-built scalar table.
    // The post-ENDPGM payload must be owned and hashed by the graphics cache in this form as well.
    std::vector<uint32_t> pcrel_smem_ps = {
        0xbe801f00u,               // s_getpc_b64 s[0:1]
        0x800000ffu, 52u,          // table = next-PC byte 4 + 52 = byte 56
        0x82010180u,               // s_addc_u32 s1, 0, s1
        0xbe820394u,               // s_mov_b32 s2, 20 bytes
        0xbe8303ffu, 0x10005004u,  // s_mov_b32 s3, V# config
        0xbeea0384u,               // s_mov_b32 vcc_lo, 4
        0xf4280100u, 0xd4000000u,  // s_buffer_load_dwordx4 s[4:7], s[0:3], vcc_lo
        0x7e000204u,               // v_mov_b32 v0, s4
        0xf800180fu, 0x00000000u,  // exp mrt0 v0,v0,v0,v0
        0xbf810000u,               // s_endpgm
        7u, 11u, 13u, 17u, 19u,
    };
    CHECK(rdna2_recompile_code_span(pcrel_smem_ps.data(), pcrel_smem_ps.size()) ==
              pcrel_smem_ps.size(),
          "PC-relative SMEM cache span includes the embedded scalar table tail");
    const auto direct_pcrel_smem_ps = recompile_fragment(
        pcrel_smem_ps.data(), pcrel_smem_ps.size(), nullptr);
    const auto cached_pcrel_smem_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, pcrel_smem_ps.data(), pcrel_smem_ps.size(), nullptr);
    CHECK(!direct_pcrel_smem_ps.empty() && cached_pcrel_smem_ps == direct_pcrel_smem_ps,
          "fragment cache retains a proven post-ENDPGM PC-relative SMEM table");
    const auto pcrel_smem_stats = shader_recompile_cache_stats();
    pcrel_smem_ps[pcrel_smem_ps.size() - 4] = 23u; // selected table[1]
    const auto changed_pcrel_smem_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, pcrel_smem_ps.data(), pcrel_smem_ps.size(), nullptr);
    stats = shader_recompile_cache_stats();
    CHECK(!changed_pcrel_smem_ps.empty() && changed_pcrel_smem_ps != cached_pcrel_smem_ps &&
              stats.misses == pcrel_smem_stats.misses + 1,
          "PC-relative SMEM table contents participate in the shader cache key");

    // A terminal if/else may build a PC-relative V# only in the second arm, after the first
    // s_endpgm. Span analysis must use the same extended instruction stream as emission. Stopping at
    // span pc20 immediately after the second end would omit four table dwords that can change SPIR-V.
    const uint32_t terminal_pcrel[] = {
        0xbe800387u, 0xbe810388u, 0xbf060100u, 0xbf840003u,
        0x7e0002ffu, 0x42280000u, 0xbf810000u,
        0xbe841f00u,               // pc7: s_getpc_b64 s[4:5] (next PC byte = 32)
        0x800404b0u,               // pc8: s_add_u32 s4, 48, s4 (table byte = 80)
        0x82050580u,               // pc9: s_addc_u32 s5, 0, s5
        0xbe860390u,               // pc10: s_mov_b32 s6, 16 bytes
        0xbe8703ffu, 0x10005004u,  // pc11: s_mov_b32 s7, V# format/stride
        0x7e020f00u,               // pc13: v_cvt_u32_f32 v1, v0
        0x34020282u,               // pc14: v_lshlrev_b32 v1, 2, v1
        0xe0301000u, 0x80010101u,  // pc15: buffer_load_dword v1, v1, s[4:7], 0 offen
        0xbf8c3f70u,               // pc17: s_waitcnt vmcnt(0)
        0x7e000d01u,               // pc18: v_cvt_f32_u32 v0, v1
        0xbf810000u,               // pc19: second s_endpgm
        7u, 11u, 13u, 17u,        // pc20: embedded table used only by the else arm
    };
    CHECK(rdna2_recompile_code_span(terminal_pcrel, std::size(terminal_pcrel)) ==
              std::size(terminal_pcrel),
          "terminal else-arm PC-relative table participates in the owning shader span");

    // Unity emits a bounded uniform jump table for uber-shader variants. The selector is loaded from
    // a direct constant buffer, adjusted by -1, clamped, scaled by eight, then used by s_setpc_b64.
    // This compact synthetic stream has two paths which set v0 differently before a common export.
    std::vector<uint32_t> dispatch_ps = {
        0xf4201a8cu, 0xfa000010u, // pc0: s_buffer_load_dword s106, s[24:27], 0x10
        0x816ac16au,              // pc2: s_add_i32 s106, s106, -1
        0x83ea826au,              // pc3: s_min_u32 s106, s106, 2
        0x8f6a836au,              // pc4: s_lshl_b32 s106, s106, 3
        0xbea01f00u,              // pc5: s_getpc_b64 s[32:33]
        0x802020ffu, 64u,         // pc6: add table byte delta (table starts at aligned pc22)
        0x82212180u,              // pc8: s_addc_u32 s33, 0, s33
        0xf4040890u, 0xd4000000u, // pc9: s_load_dwordx2 s[34:35], s[32:33], s106
        0xbea81f00u,              // pc11: s_getpc_b64 s[40:41]
        0x80282228u,              // pc12: s_add_u32 s40, s40, s34
        0x82292329u,              // pc13: s_addc_u32 s41, s41, s35
        0xbe802028u,              // pc14: s_setpc_b64 s[40:41]
        0x7e000280u,              // pc15 target A: v_mov_b32 v0, 0
        0xbf820001u,              // pc16: s_branch common export at pc18
        0x7e0002f2u,              // pc17 target B: v_mov_b32 v0, 1.0
        0xf800180fu, 0x00000000u, // pc18: exp mrt0 v0,v0,v0,v0
        0xbf810000u,              // pc20: s_endpgm
        0u,                       // pc21: alignment padding before the qword table
    };
    // Entries are signed byte offsets relative to the instruction after the second s_getpc (pc12).
    for (uint32_t index = 0; index < 3; ++index) {
        dispatch_ps.push_back(index == 0 ? 12u : index == 1 ? 20u : 24u); // pc15 / pc17 / merge pc18
        dispatch_ps.push_back(0u);
    }
    const PcrelDispatchInfo dispatch = rdna2_pcrel_dispatch_info(
        dispatch_ps.data(), dispatch_ps.size());
    CHECK(dispatch.valid && dispatch.selector_sgpr_base == 24 &&
              dispatch.selector_byte_offset == 0x10 && dispatch.selector_addend == -1 &&
              dispatch.selector_max == 2 && dispatch.target_pcs.size() == 3 &&
              dispatch.target_pcs[0] == 15 && dispatch.target_pcs[1] == 17 &&
              dispatch.target_pcs[2] == 18,
          "bounded PC-relative scalar dispatch is recognized and every target is proven");
    CHECK(rdna2_recompile_code_span(dispatch_ps.data(), dispatch_ps.size()) == dispatch_ps.size(),
          "scalar-dispatch table tail participates in the owning shader span");

    std::vector<uint32_t> reversed_shift_dispatch = dispatch_ps;
    reversed_shift_dispatch[4] = 0x8f6a6a83u; // s_lshl_b32 s106, 3, s106
    CHECK(!rdna2_pcrel_dispatch_info(reversed_shift_dispatch.data(),
                                     reversed_shift_dispatch.size()).valid,
          "reversed non-commutative dispatch shift is rejected");

    std::array<uint32_t, 5> selector_words{};
    ShaderResourceTable dispatch_table;
    ShaderResource selector_resource;
    selector_resource.cls = ResourceClass::ConstantBuffer;
    selector_resource.binding = 0;
    selector_resource.sgpr_base = 24;
    selector_resource.size = sizeof(selector_words);
    selector_resource.host_data = reinterpret_cast<uint8_t*>(selector_words.data());
    selector_resource.host_data_size = sizeof(selector_words);
    dispatch_table.resources.push_back(selector_resource);
    const auto direct_dispatch_a = recompile_fragment(
        dispatch_ps.data(), dispatch_ps.size(), &dispatch_table, nullptr, 15);
    const auto direct_dispatch_b = recompile_fragment(
        dispatch_ps.data(), dispatch_ps.size(), &dispatch_table, nullptr, 17);
    CHECK(!direct_dispatch_a.empty() && !direct_dispatch_b.empty(),
          "direct fragment specialization accepts both proven dispatch paths");
    selector_words[4] = 1; // adjusted selector 0 -> target A
    const auto dispatch_a = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, dispatch_ps.data(), dispatch_ps.size(), &dispatch_table);
    const auto dispatch_stats = shader_recompile_cache_stats();
    selector_words[4] = 2; // adjusted selector 1 -> target B
    const auto dispatch_b = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, dispatch_ps.data(), dispatch_ps.size(), &dispatch_table);
    stats = shader_recompile_cache_stats();
    CHECK(!dispatch_a.empty() && !dispatch_b.empty(),
          "both proven uniform-dispatch targets recompile");
    CHECK(dispatch_a != dispatch_b,
          "uniform dispatch target specializes the emitted fragment code");
    CHECK(stats.misses == dispatch_stats.misses + 1,
          "uniform dispatch target participates in the cache key");

    // Interpolant wiring changes vertex PARAM export locations/defaults, so it is part of the
    // compile-time key even when the guest code and resource interface are otherwise identical.
    stats = shader_recompile_cache_stats();
    const uint64_t mapping_misses = stats.misses;
    const uint64_t mapping_hits = stats.hits;
    PixelInputMapping mapping;
    mapping.valid_mask = 1;
    mapping.controls[0] = 1;
    const auto mapped_once = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, &mapping);
    const auto mapped_again = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, &mapping);
    mapping.controls[0] = 2;
    const auto remapped = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, &mapping);
    stats = shader_recompile_cache_stats();
    CHECK(!mapped_once.empty() && mapped_again == mapped_once && !remapped.empty() &&
              stats.misses == mapping_misses + 2 && stats.hits == mapping_hits + 1,
          "pixel-input mappings participate in the vertex shader cache key");

    // The same mapping also selects the fragment-side interpolation interface. In particular,
    // custom passthrough inputs consume raw triangle values instead of synthesized P10/P20 deltas.
    static const uint32_t kInterpPs[] = {
        0xc80e0002u, 0xc8110002u, 0xf800000fu, 0x03030303u, 0xbf810000u,
    };
    const uint64_t fragment_mapping_misses = stats.misses;
    PixelInputMapping smooth_mapping;
    smooth_mapping.valid_mask = 1;
    const auto smooth_fragment = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kInterpPs, std::size(kInterpPs), nullptr,
        &smooth_mapping);
    PixelInputMapping passthrough_mapping = smooth_mapping;
    passthrough_mapping.passthrough_mask = 1;
    const auto passthrough_fragment = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kInterpPs, std::size(kInterpPs), nullptr,
        &passthrough_mapping);
    const auto passthrough_again = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kInterpPs, std::size(kInterpPs), nullptr,
        &passthrough_mapping);
    stats = shader_recompile_cache_stats();
    CHECK(!smooth_fragment.empty() && !passthrough_fragment.empty() &&
              passthrough_again == passthrough_fragment &&
              stats.misses == fragment_mapping_misses + 2,
          "pixel-input mappings participate in the fragment interpolation cache key");

    // Fragment system-input placement changes SPIR-V declarations and the initial VGPR values.
    // Both ENA and ADDR therefore belong to the cache key.
    const uint64_t system_misses = stats.misses;
    PixelSystemInputMapping system_inputs{0x00000303u, 0x00000303u};
    const auto system_once = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr, nullptr, &system_inputs);
    const auto system_again = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr, nullptr, &system_inputs);
    system_inputs.addr |= 1u << 2;
    const auto system_remapped = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr, nullptr, &system_inputs);
    stats = shader_recompile_cache_stats();
    CHECK(!system_once.empty() && system_again == system_once && !system_remapped.empty() &&
              stats.misses == system_misses + 2,
          "pixel-system ENA/ADDR mappings participate in the fragment shader cache key");

    // SPI_PS_IN_CONTROL.PS_W32_EN changes low-half EXEC/VCC semantics, so the same program and
    // descriptor interface must not reuse a Wave64/default module under Wave32 (or vice versa).
    const uint64_t wave_mode_hits = stats.hits;
    const uint64_t wave_mode_misses = stats.misses;
    uint64_t wave64_identity = 0, wave32_identity = 0, repeated_wave32_identity = 0;
    const auto wave64_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr, nullptr, nullptr,
        &wave64_identity, false);
    const auto wave32_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr, nullptr, nullptr,
        &wave32_identity, true);
    const auto repeated_wave32_ps = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kPs, std::size(kPs), nullptr, nullptr, nullptr,
        &repeated_wave32_identity, true);
    stats = shader_recompile_cache_stats();
    CHECK(!wave64_ps.empty() && wave32_ps == wave64_ps && repeated_wave32_ps == wave32_ps &&
              wave64_identity != 0 && wave32_identity != 0 &&
              wave32_identity != wave64_identity && repeated_wave32_identity == wave32_identity &&
              stats.hits == wave_mode_hits + 2 && stats.misses == wave_mode_misses + 1,
          "fragment Wave32 mode participates in the shader cache key");

    const uint64_t identity_before_clear = first_identity;
    clear_shader_recompile_cache();
    stats = shader_recompile_cache_stats();
    CHECK(stats.entries == 0 && stats.hits == 0 && stats.misses == 0 && stats.bytes == 0,
          "cache reset clears entries and instrumentation");
    uint64_t identity_after_clear = 0;
    (void)recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &identity_after_clear);
    CHECK(identity_after_clear > identity_before_clear,
          "cache reset never recycles compiled-shader identities");

    clear_shader_recompile_cache();
    clear_shader_analysis_cache();
    std::vector<uint32_t> mutable_ps(kPs, kPs + std::size(kPs));
    (void)recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, mutable_ps.data(), mutable_ps.size(), nullptr);
    (void)recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, mutable_ps.data(), mutable_ps.size(), nullptr);
    mutable_ps[0] ^= 1u;  // same allocation, changed instruction bytes
    (void)recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, mutable_ps.data(), mutable_ps.size(), nullptr);
    analysis_stats = shader_analysis_cache_stats();
    stats = shader_recompile_cache_stats();
    CHECK(analysis_stats.hits == 1 && analysis_stats.misses == 2 &&
              analysis_stats.invalidations == 1 && stats.hits == 1 && stats.misses == 2,
          "same-address shader mutation invalidates analysis and misses the compiled cache");

    // Live realization retains the cache allocation directly. Repeated hits must share one immutable
    // word vector, and eviction/reset must not invalidate a DrawItem that still owns that version.
    clear_shader_recompile_cache();
    uint64_t shared_first_identity = 0, shared_second_identity = 0;
    const SharedShaderWords shared_first = recompile_graphics_shader_cached_shared(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &shared_first_identity);
    const SharedShaderWords shared_second = recompile_graphics_shader_cached_shared(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &shared_second_identity);
    stats = shader_recompile_cache_stats();
    CHECK(shared_first && shared_first == shared_second && *shared_first ==
              recompile_vertex(kVs, std::size(kVs), &table) &&
              shared_first_identity != 0 && shared_first_identity == shared_second_identity &&
              stats.misses == 1 && stats.hits == 1,
          "live shader-cache hits share one exact immutable SPIR-V allocation");
    clear_shader_recompile_cache();
    CHECK(shared_first && !shared_first->empty(),
          "shared shader words outlive cache reset while a realized draw retains them");

    // Compute realization used to run the full RDNA2 walk/CFG/SPIR-V builder for every dispatch,
    // even though SGPR values are supplied as push constants. Cache by code, descriptor interface,
    // and the launch ABI fields that actually specialize the module.
    static const uint32_t kCompute[] = {
        0xd7460004, 0x04010c08, 0x7e000204, 0x7e020205, 0x7e040206,
        0x7e060207, 0xe01c2000, 0x80000004, 0xbf810000,
    };
    ShaderResource compute_buffer;
    compute_buffer.cls = ResourceClass::ConstantBuffer;
    compute_buffer.format = DataFormat::Uint32;
    compute_buffer.num_components = 4;
    compute_buffer.binding = 2;
    compute_buffer.stride = 16;
    compute_buffer.sgpr_base = 0;
    compute_buffer.size = 4096;
    ShaderResourceTable compute_table;
    compute_table.resources.push_back(compute_buffer);
    ComputeShaderConfig compute_config;
    compute_config.user_sgprs.resize(8);
    compute_config.local_x = 64;
    compute_config.exact_thread_extent = true;
    compute_config.threads_x = 130;
    compute_config.threads_y = compute_config.threads_z = 1;
    compute_config.tgid_x_en = true;
    const RecompileDiagnosticContext diagnostic_a{
        RecompileDiagnosticStage::Compute, 0x1111222233334444ull};
    const RecompileDiagnosticContext diagnostic_b{
        RecompileDiagnosticStage::Compute, 0xaaaabbbbccccddddull};
    const auto direct_compute = recompile_compute(
        kCompute, std::size(kCompute), &compute_table, compute_config);
    uint64_t cached_compute_identity = 0, relocated_diagnostic_identity = 0,
             push_constant_identity = 0;
    const auto cached_compute = recompile_compute_shader_cached(
        kCompute, std::size(kCompute), &compute_table, compute_config,
        &cached_compute_identity, diagnostic_a);
    const auto relocated_diagnostic_compute = recompile_compute_shader_cached(
        kCompute, std::size(kCompute), &compute_table, compute_config,
        &relocated_diagnostic_identity, diagnostic_b);
    ComputeShaderConfig new_push_constants = compute_config;
    new_push_constants.user_sgprs[4] = 0x12345678;
    const auto repeated_compute = recompile_compute_shader_cached(
        kCompute, std::size(kCompute), &compute_table, new_push_constants,
        &push_constant_identity, diagnostic_b);
    ComputeShaderConfig changed_launch = compute_config;
    changed_launch.local_x = 32;
    const auto changed_compute = recompile_compute_shader_cached(
        kCompute, std::size(kCompute), &compute_table, changed_launch);
    stats = shader_recompile_cache_stats();
    CHECK(!direct_compute.empty() && cached_compute == direct_compute &&
              relocated_diagnostic_compute == direct_compute && repeated_compute == direct_compute &&
              !changed_compute.empty() && changed_compute != direct_compute &&
              stats.misses == 2 && stats.hits == 2,
          "compute cache reuses push-constant variants and separates launch-specialized modules");
    CHECK(cached_compute_identity != 0 &&
              relocated_diagnostic_identity == cached_compute_identity &&
              push_constant_identity == cached_compute_identity,
          "diagnostic provenance stays outside compute shader cache identity");

    // BUFFER_ATOMIC_FMIN bakes COMPUTE_PGM_RSRC1.FP32_DENORM into its integer CAS selection.
    // The same packet must therefore miss across launch modes, while repeating one mode hits.
    clear_shader_recompile_cache();
    static const uint32_t kAtomicFminCompute[] = {
        0xe0fc0000u, 0x80010000u, // buffer_atomic_fmin v0, off, s[4:7], 0
        0xbf810000u,
    };
    ShaderResource atomic_buffer;
    atomic_buffer.cls = ResourceClass::ConstantBuffer;
    atomic_buffer.format = DataFormat::Uint32;
    atomic_buffer.num_components = 1;
    atomic_buffer.binding = 3;
    atomic_buffer.stride = 4;
    atomic_buffer.sgpr_base = 4;
    atomic_buffer.size = 4;
    ShaderResourceTable float_atomic_table;
    float_atomic_table.resources.push_back(atomic_buffer);
    ComputeShaderConfig atomic_mode0 = compute_config;
    atomic_mode0.compute_pgm_rsrc1 = 0u;
    ComputeShaderConfig atomic_mode3 = atomic_mode0;
    atomic_mode3.compute_pgm_rsrc1 = kDefaultComputePgmRsrc1;
    const auto atomic_flush = recompile_compute_shader_cached(
        kAtomicFminCompute, std::size(kAtomicFminCompute), &float_atomic_table, atomic_mode0);
    const auto atomic_preserve = recompile_compute_shader_cached(
        kAtomicFminCompute, std::size(kAtomicFminCompute), &float_atomic_table, atomic_mode3);
    const auto atomic_flush_again = recompile_compute_shader_cached(
        kAtomicFminCompute, std::size(kAtomicFminCompute), &float_atomic_table, atomic_mode0);
    stats = shader_recompile_cache_stats();
    CHECK(!atomic_flush.empty() && !atomic_preserve.empty() &&
              atomic_flush != atomic_preserve && atomic_flush_again == atomic_flush &&
              stats.misses == 2 && stats.hits == 1 && stats.entries == 2,
          "compute cache separates float-atomic modules by PGM_RSRC1 denormal mode");

    // Diagnostic provenance is not shader semantics. Force two recompiles of one rejected site so
    // each call must carry its own immutable program identity through the cache wrapper, builder and
    // structurizer, while a direct standalone call must remain explicitly neutral.
    static const uint32_t kRejectedCompute[] = {
        0xbf890000u, // s_cbranch_execnz with target pc1: unclaimed outside a recognized loop
        0xbf810000u, // s_endpgm
    };
    set_test_env("PROSPER_DBG", "1");
    set_test_env("PROSPER_NO_SHADER_CACHE", "1");
    const CapturedStderr diagnostic_capture = capture_stderr([&] {
        (void)recompile_compute_shader_cached(
            kRejectedCompute, std::size(kRejectedCompute), nullptr, compute_config,
            nullptr, diagnostic_a);
        (void)recompile_compute_shader_cached(
            kRejectedCompute, std::size(kRejectedCompute), nullptr, compute_config,
            nullptr, diagnostic_b);
        (void)recompile_compute(kRejectedCompute, std::size(kRejectedCompute), nullptr,
                                compute_config);
    });
    const std::string& diagnostic_log = diagnostic_capture.output;
    set_test_env("PROSPER_NO_SHADER_CACHE", nullptr);
    set_test_env("PROSPER_DBG", nullptr);
    std::error_code diagnostic_capture_error;
    CHECK(diagnostic_capture.path.parent_path() == prosper_test::test_scratch_dir() &&
              !std::filesystem::exists(diagnostic_capture.path, diagnostic_capture_error) &&
              !diagnostic_capture_error,
          "diagnostic capture uses and cleans the process-private test scratch directory");
    CHECK(diagnostic_log.find(
              "[compute-struct-reject] unclaimed execnz pc=0 stage=compute "
              "program=0x1111222233334444 role=route-decline\n") != std::string::npos &&
              diagnostic_log.find(
              "[compute-struct-reject] unclaimed execnz pc=0 stage=compute "
              "program=0xaaaabbbbccccdddd role=route-decline\n") != std::string::npos,
          "distinct compute calls retain their own program identity at the structurizer site");
    CHECK(diagnostic_log.find(
              " stage=compute program=0x1111222233334444 role=terminal\n") !=
              std::string::npos &&
              diagnostic_log.find(
              " stage=compute program=0xaaaabbbbccccdddd role=terminal\n") !=
              std::string::npos,
          "terminal instruction rejects retain the same per-call diagnostic identity");
    CHECK(diagnostic_log.find(
              "[compute-struct-reject] unclaimed execnz pc=0 stage=compute "
              "program=none role=route-decline\n") !=
              std::string::npos &&
              diagnostic_log.find(
              " stage=compute program=none role=terminal\n") !=
              std::string::npos,
          "standalone compute recompilation uses an explicit neutral program identity");

    // A nested guest-wave branch containing s_barrier cannot use either the compact structurizer or
    // the synchronized CFG dispatcher. This is the exact GTA V route that used to return an empty
    // module without a terminal line. Cache the rejection, then report two distinct live program
    // addresses: the second compile must be a cache hit, while the final skip consequence remains
    // attributable and once-per-address at the production reporting boundary.
    static const uint32_t kExactWaveBarrierReject[] = {
        0x7c020300u, // 0: v_cmp_lt_f32 vcc, v0, v1
        0xbf860001u, // 1: s_cbranch_vccz +1 -> pc3 (exact guest-wave branch)
        0x7e040281u, // 2: v_mov_b32 v2, 1
        0xbf060000u, // 3: s_cmp_eq_u32 s0, s0
        0xbf840002u, // 4: s_cbranch_scc0 +2 -> pc7
        0xbf8a0000u, // 5: s_barrier nested in the scalar arm
        0x7e040282u, // 6: v_mov_b32 v2, 2
        0xbf810000u, // 7: s_endpgm
    };
    // A partial workgroup whose barrier cannot be lifted into uniform control flow.
    //
    // This was `{ s_barrier, s_endpgm }` -- a partial workgroup with ANY barrier used to reject, so
    // the shortest such program served as the vehicle. That combination now compiles: the stream is
    // split at its barriers and each barrier-free phase runs under the dispatcher's ACTIVE bit, with
    // the barrier emitted at function scope. Keeping the old bytes here would have quietly turned an
    // assertion about the diagnostic plumbing into an assertion that nothing had changed.
    //
    // The branch at pc1 jumps from before the barrier to after it, so no phase boundary is legal at
    // pc2 and the split is refused -- which is a real and permanent limit, not a missing lowering.
    // The reject, its reason string, and the extents below are therefore unchanged.
    static const uint32_t kPartialBarrierReject[] = {
        0xbf060000u, // 0: s_cmp_eq_u32 s0, s0
        0xbf840002u, // 1: s_cbranch_scc0 +2 -> pc4, crossing the barrier
        0xbf8a0000u, // 2: s_barrier
        0x7e040282u, // 3: v_mov_b32 v2, 2
        0x7e040281u, // 4: v_mov_b32 v2, 1
        0xbf810000u, // 5: s_endpgm
    };
    ComputeShaderConfig rejected_config;
    rejected_config.local_x = 64;
    rejected_config.wave_size = 64;
    rejected_config.threads_x = 64;
    rejected_config.threads_y = rejected_config.threads_z = 1;
    rejected_config.exact_thread_extent = true;
    ComputeShaderConfig partial_barrier_config = rejected_config;
    partial_barrier_config.threads_x = 65;
    const RecompileDiagnosticContext rejected_diagnostic_a{
        RecompileDiagnosticStage::Compute, 0x1234567800004455ull};
    const RecompileDiagnosticContext rejected_diagnostic_b{
        RecompileDiagnosticStage::Compute, 0xabcdef120000dd55ull};
    const RecompileDiagnosticContext partial_barrier_diagnostic{
        RecompileDiagnosticStage::Compute, 0x135724680000aa55ull};
    clear_shader_recompile_cache();
    std::vector<uint32_t> rejected_first, rejected_cached, partial_barrier;
    set_test_env("PROSPER_DBG", "1");
    const CapturedStderr final_reject_capture = capture_stderr([&] {
        rejected_first = recompile_compute_shader_cached(
            kExactWaveBarrierReject, std::size(kExactWaveBarrierReject), nullptr,
            rejected_config, nullptr, rejected_diagnostic_a);
        if (rejected_first.empty())
            (void)report_compute_recompile_skip_once(rejected_diagnostic_a);
        rejected_cached = recompile_compute_shader_cached(
            kExactWaveBarrierReject, std::size(kExactWaveBarrierReject), nullptr,
            rejected_config, nullptr, rejected_diagnostic_b);
        if (rejected_cached.empty()) {
            (void)report_compute_recompile_skip_once(rejected_diagnostic_b);
            (void)report_compute_recompile_skip_once(rejected_diagnostic_b);
        }
        partial_barrier = recompile_compute(
            kPartialBarrierReject, std::size(kPartialBarrierReject), nullptr,
            partial_barrier_config, partial_barrier_diagnostic);
    });
    set_test_env("PROSPER_DBG", nullptr);
    stats = shader_recompile_cache_stats();
    const std::string& final_reject_log = final_reject_capture.output;
    std::error_code final_reject_capture_error;
    CHECK(final_reject_capture.path.parent_path() == prosper_test::test_scratch_dir() &&
              !std::filesystem::exists(final_reject_capture.path, final_reject_capture_error) &&
              !final_reject_capture_error,
          "final-reject capture uses and cleans the process-private test scratch directory");
    CHECK(rejected_first.empty() && rejected_cached.empty() &&
              stats.misses == 1 && stats.hits == 1,
          "rejected compute modules retain empty results across the real shader cache boundary");
    CHECK(count_occurrences(
              final_reject_log,
              "[compute-cfg-reject] reason=exact-wave-dispatcher-unsafe guest-barrier=1 "
              "stage=compute program=0x1234567800004455 role=terminal\n") == 1 &&
              final_reject_log.find(
              "reason=exact-wave-dispatcher-unsafe guest-barrier=1 stage=compute "
              "program=0xabcdef120000dd55 role=terminal") == std::string::npos,
          "exact-wave guest-barrier rejection is terminal at the compiler site and not replayed by a cache hit");
    CHECK(count_occurrences(
              final_reject_log,
              "[compute-recompile-reject] reason=empty-result dispatch-skipped") == 2 &&
              count_occurrences(
              final_reject_log,
              "stage=compute program=0x1234567800004455 role=consequent\n") == 1 &&
              count_occurrences(
              final_reject_log,
              "stage=compute program=0xabcdef120000dd55 role=consequent\n") == 1,
          "live empty-result consequences cover compile misses and cache hits once per program address");
    CHECK(partial_barrier.empty() && count_occurrences(
              final_reject_log,
              "[compute-recompile-reject] reason=partial-workgroup-barrier "
              "threads=65x1x1 local=64x1x1 stage=compute "
              "program=0x135724680000aa55 role=terminal\n") == 1,
          "partial-workgroup barrier rejection reports its exact terminal cause and program identity");

    const RecompileDiagnosticContext legacy_skip_diagnostic{
        RecompileDiagnosticStage::Compute, 0x246813570000bb55ull};
    const CapturedStderr legacy_skip_capture = capture_stderr([&] {
        (void)report_compute_recompile_skip_once(legacy_skip_diagnostic);
        (void)report_compute_recompile_skip_once(legacy_skip_diagnostic);
    });
    CHECK(count_occurrences(
              legacy_skip_capture.output,
              "[compute] skip unsupported program 0x246813570000bb55 reason=unrecorded\n") == 1 &&
              legacy_skip_capture.output.find("role=consequent") == std::string::npos,
          "non-debug live skips stay once-per-address and say so when no reason was recorded");

    // The point of the reason field: a program whose reject WAS recorded must name it on the
    // non-debug line. Without this arm the change is indistinguishable from appending a constant --
    // `reason=unrecorded` alone is satisfied by a version that never looks the reason up.
    const uint64_t reasoned_program = 0x246813570000cc55ull;
    const RecompileDiagnosticContext reasoned_diagnostic{
        RecompileDiagnosticStage::Compute, reasoned_program};
    (void)capture_stderr([&] {
        record_recompile_reject_reason_for_test(reasoned_diagnostic, "compute-struct-reject",
                                          "route-decline", "unclaimed execnz pc=428");
    });
    const CapturedStderr reasoned_skip_capture = capture_stderr([&] {
        (void)report_compute_recompile_skip_once(reasoned_diagnostic);
    });
    CHECK(count_occurrences(
              reasoned_skip_capture.output,
              "[compute] skip unsupported program 0x246813570000cc55 "
              "reason=compute-struct-reject unclaimed execnz pc=428\n") == 1,
          "a recorded reject reason reaches the non-debug skip line");

    // And a "consequent" line must NOT overwrite it. A consequent only restates that an empty result
    // was returned; if it could win, the diagnostic would replace the answer with the question.
    (void)capture_stderr([&] {
        record_recompile_reject_reason_for_test(reasoned_diagnostic, "compute-recompile-reject",
                                          "consequent", "reason=empty-result dispatch-skipped");
    });
    CHECK(last_terminal_reject_reason(reasoned_program) ==
              "compute-struct-reject unclaimed execnz pc=428",
          "a consequent line does not overwrite the recorded cause");

    // The unit-level reporter checks above deliberately keep cache-hit and legacy behavior focused,
    // but they cannot guard the executor's integration site: a live shader-recompile failure must
    // actually call that reporter before it discards the dispatch. Register a rejected program and
    // drive it through the complete realization boundary so deleting the production call loses this
    // consequent even though the compiler's terminal diagnostic and the helper itself still work.
    prosper::register_agc_hle();
    auto create_shader = prosper::Hle::lookup("f3dg2CSgRKY");
    // Same shape, and for the same reason, as kPartialBarrierReject above: a barrier the phase split
    // cannot lift, because a guest branch crosses it.
    alignas(256) static const uint32_t kLivePartialBarrierReject[] = {
        0xbf060000u, // 0: s_cmp_eq_u32 s0, s0
        0xbf840002u, // 1: s_cbranch_scc0 +2 -> pc4, crossing the barrier
        0xbf8a0000u, // 2: s_barrier
        0x7e040282u, // 3: v_mov_b32 v2, 2
        0x7e040281u, // 4: v_mov_b32 v2, 1
        0xbf810000u, // 5: s_endpgm
    };
    ShaderReg live_reject_registers[2] = {
        {prosper::agc::Pm4::COMPUTE_PGM_LO, 0},
        {prosper::agc::Pm4::COMPUTE_PGM_HI, 0},
    };
    AgcShaderHeader live_reject_header{};
    live_reject_header.file_header = 0x34333231u;
    live_reject_header.version = 0x18;
    live_reject_header.sh_registers = live_reject_registers;
    live_reject_header.shader_size = sizeof(kLivePartialBarrierReject);
    live_reject_header.type = 0;
    live_reject_header.num_sh_registers = 2;
    void* registered_live_reject = nullptr;
    const bool live_reject_registered = create_shader &&
        create_shader(reinterpret_cast<uint64_t>(&registered_live_reject),
                      reinterpret_cast<uint64_t>(&live_reject_header),
                      reinterpret_cast<uint64_t>(kLivePartialBarrierReject), 0, 0, 0) == 0 &&
        registered_live_reject == &live_reject_header;
    CHECK(live_reject_registered,
          "register a rejected compute program for the live realization boundary");

    GpuState live_reject_state;
    live_reject_state.sh[prosper::agc::Pm4::COMPUTE_PGM_LO] = live_reject_registers[0].value;
    live_reject_state.sh[prosper::agc::Pm4::COMPUTE_PGM_HI] = live_reject_registers[1].value;
    live_reject_state.sh[prosper::agc::Pm4::COMPUTE_NUM_THREAD_X] = 64;
    live_reject_state.sh[prosper::agc::Pm4::COMPUTE_NUM_THREAD_Y] = 1;
    live_reject_state.sh[prosper::agc::Pm4::COMPUTE_NUM_THREAD_Z] = 1;
    GpuState::Dispatch live_reject_dispatch;
    live_reject_dispatch.threads_x = 65;
    live_reject_dispatch.threads_y = live_reject_dispatch.threads_z = 1;
    live_reject_dispatch.modifier =
        1ull << prosper::agc::Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_SHIFT;
    live_reject_dispatch.command_order = 0x2481;
    live_reject_dispatch.state = std::make_shared<GpuState>(live_reject_state);
    live_reject_state.dispatches.push_back(live_reject_dispatch);

    std::vector<ComputeItem> live_reject_items;
    std::vector<OperationRealizationFailure> live_reject_failures;
    set_test_env("PROSPER_DBG", "1");
    const CapturedStderr live_reject_capture = capture_stderr([&] {
        live_reject_items = realize_compute_dispatches(
            live_reject_state, 0x2481, &live_reject_failures);
    });
    set_test_env("PROSPER_DBG", nullptr);
    char live_reject_program[96];
    std::snprintf(live_reject_program, sizeof(live_reject_program),
                  "stage=compute program=0x%llx role=consequent\n",
                  static_cast<unsigned long long>(
                      reinterpret_cast<uint64_t>(kLivePartialBarrierReject)));
    CHECK(live_reject_registered && live_reject_items.empty() &&
              live_reject_failures.size() == 1 &&
              live_reject_failures[0].reason == RealizationFailureReason::ShaderRecompile &&
              live_reject_failures[0].stages.size() == 1 &&
              live_reject_failures[0].stages[0].program_addr ==
                  reinterpret_cast<uint64_t>(kLivePartialBarrierReject),
          "registered rejected compute reaches the live shader-recompile failure boundary");
    CHECK(count_occurrences(
              live_reject_capture.output,
              "[compute-recompile-reject] reason=empty-result dispatch-skipped") == 1 &&
              count_occurrences(live_reject_capture.output, live_reject_program) == 1,
          "live shader-recompile failure reports its final dispatch-skipped consequence");

    // Drive FP32_DENORM through the actual register-file realization boundary. Unit-level config
    // and cache tests cannot catch a missing/wrong register read here: deleting the production read
    // would leave both dispatches at the synthetic mode-3 default and make these modules identical.
    alignas(256) static const uint32_t kLiveFloatAtomicMode[] = {
        0x7e000280u,              // v_mov_b32 v0, 0 (LDS byte address)
        0x7e1202ffu, 0x00000001u, // v_mov_b32 v9, +minimum subnormal
        0xd8480000u, 0x00000900u, // ds_min_f32 v0, v9
        0xbf810000u,
    };
    ShaderReg live_float_registers[2] = {
        {prosper::agc::Pm4::COMPUTE_PGM_LO, 0},
        {prosper::agc::Pm4::COMPUTE_PGM_HI, 0},
    };
    AgcShaderHeader live_float_header{};
    live_float_header.file_header = 0x34333231u;
    live_float_header.version = 0x18;
    live_float_header.sh_registers = live_float_registers;
    live_float_header.shader_size = sizeof(kLiveFloatAtomicMode);
    live_float_header.type = 0;
    live_float_header.num_sh_registers = 2;
    void* registered_live_float = nullptr;
    const bool live_float_registered = create_shader &&
        create_shader(reinterpret_cast<uint64_t>(&registered_live_float),
                      reinterpret_cast<uint64_t>(&live_float_header),
                      reinterpret_cast<uint64_t>(kLiveFloatAtomicMode), 0, 0, 0) == 0 &&
        registered_live_float == &live_float_header;
    auto realize_float_mode = [&](uint32_t compute_pgm_rsrc1, uint64_t command_order) {
        GpuState state;
        state.sh[prosper::agc::Pm4::COMPUTE_PGM_LO] = live_float_registers[0].value;
        state.sh[prosper::agc::Pm4::COMPUTE_PGM_HI] = live_float_registers[1].value;
        state.sh[prosper::agc::Pm4::COMPUTE_PGM_RSRC1] = compute_pgm_rsrc1;
        state.sh[prosper::agc::Pm4::COMPUTE_NUM_THREAD_X] = 64;
        state.sh[prosper::agc::Pm4::COMPUTE_NUM_THREAD_Y] = 1;
        state.sh[prosper::agc::Pm4::COMPUTE_NUM_THREAD_Z] = 1;
        GpuState::Dispatch dispatch;
        dispatch.threads_x = 64;
        dispatch.threads_y = dispatch.threads_z = 1;
        dispatch.modifier =
            1ull << prosper::agc::Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_SHIFT;
        dispatch.command_order = command_order;
        dispatch.state = std::make_shared<GpuState>(state);
        state.dispatches.push_back(dispatch);
        return realize_compute_dispatches(state, 0x2481);
    };
    clear_shader_recompile_cache();
    const std::vector<ComputeItem> live_float_mode0 = realize_float_mode(0u, 0x24810u);
    const std::vector<ComputeItem> live_float_mode3 = realize_float_mode(
        kDefaultComputePgmRsrc1, 0x24813u);
    CHECK(live_float_registered && live_float_mode0.size() == 1 &&
              live_float_mode3.size() == 1 &&
              live_float_mode0[0].recompile_config.compute_pgm_rsrc1 == 0u &&
              live_float_mode3[0].recompile_config.compute_pgm_rsrc1 ==
                  kDefaultComputePgmRsrc1 &&
              live_float_mode0[0].spirv != live_float_mode3[0].spirv,
          "live compute realization reads PGM_RSRC1 and separates float-atomic denormal modes");

    // IMAGE_LOAD_MIP level-zero lowering is compiled semantics, not runtime descriptor data. A
    // captured or later dispatch without the instruction-scoped proof must miss and reject rather
    // than reuse the specialized module; restoring the proof should hit the original entry.
    clear_shader_recompile_cache();
    static const uint32_t kZeroMipCompute[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7
        0xf0043f08u, 0x00050000u,            // IMAGE_LOAD_MIP xyzw 2D, T# s[20:27]
        0xbf810000u,
    };
    ShaderResource zero_mip_texture;
    zero_mip_texture.cls = ResourceClass::Texture;
    zero_mip_texture.format = DataFormat::Uint32;
    zero_mip_texture.num_components = 1;
    zero_mip_texture.binding = 4;
    zero_mip_texture.fetch_pc = 1;
    zero_mip_texture.img_dim = 1;
    zero_mip_texture.width = zero_mip_texture.height = 4;
    zero_mip_texture.depth = 1;
    zero_mip_texture.size = 64;
    zero_mip_texture.proven_zero_mip = true;
    ShaderResourceTable zero_mip_table;
    zero_mip_table.resources.push_back(zero_mip_texture);
    ComputeShaderConfig zero_mip_config;
    zero_mip_config.user_sgprs.resize(28);
    zero_mip_config.local_x = 1;
    const auto cached_proven_zero_mip = recompile_compute_shader_cached(
        kZeroMipCompute, std::size(kZeroMipCompute), &zero_mip_table, zero_mip_config);
    zero_mip_table.resources[0].proven_zero_mip = false;
    const auto cached_unproven_zero_mip = recompile_compute_shader_cached(
        kZeroMipCompute, std::size(kZeroMipCompute), &zero_mip_table, zero_mip_config);
    zero_mip_table.resources[0].proven_zero_mip = true;
    const auto repeated_proven_zero_mip = recompile_compute_shader_cached(
        kZeroMipCompute, std::size(kZeroMipCompute), &zero_mip_table, zero_mip_config);
    zero_mip_table.resources[0].declared_mip_levels = 2;
    const auto cached_multilevel_zero_mip = recompile_compute_shader_cached(
        kZeroMipCompute, std::size(kZeroMipCompute), &zero_mip_table, zero_mip_config);
    zero_mip_table.resources[0].declared_mip_levels = 1;
    zero_mip_table.resources[0].compression_enabled = true;
    const auto cached_dcc_zero_mip = recompile_compute_shader_cached(
        kZeroMipCompute, std::size(kZeroMipCompute), &zero_mip_table, zero_mip_config);
    zero_mip_table.resources[0].compression_enabled = false;
    zero_mip_table.resources[0].in_mip_tail = true;
    const auto cached_tail_zero_mip = recompile_compute_shader_cached(
        kZeroMipCompute, std::size(kZeroMipCompute), &zero_mip_table, zero_mip_config);
    stats = shader_recompile_cache_stats();
    CHECK(!cached_proven_zero_mip.empty() && cached_unproven_zero_mip.empty() &&
              repeated_proven_zero_mip == cached_proven_zero_mip &&
              cached_multilevel_zero_mip.empty() && cached_dcc_zero_mip.empty() &&
              cached_tail_zero_mip.empty() &&
              stats.misses == 5 && stats.hits == 1 && stats.entries == 5,
          "compute cache partitions zero-mip proof, levels, DCC, and texture-tail safety");

    clear_shader_recompile_cache();
    static const uint32_t kZeroStoreMipCompute[] = {
        0x7e0a0206u,
        0xf024310au, 0x00030004u, 0x00000503u,
        0xbf810000u,
    };
    ShaderResource zero_store_mip_image;
    zero_store_mip_image.cls = ResourceClass::StorageImage;
    zero_store_mip_image.format = DataFormat::Uint32;
    zero_store_mip_image.num_components = 1;
    zero_store_mip_image.binding = 4;
    zero_store_mip_image.fetch_pc = 1;
    zero_store_mip_image.img_dim = 1;
    zero_store_mip_image.width = zero_store_mip_image.height = 4;
    zero_store_mip_image.depth = 1;
    zero_store_mip_image.size = 64;
    zero_store_mip_image.proven_zero_mip = true;
    ShaderResourceTable zero_store_mip_table;
    zero_store_mip_table.resources.push_back(zero_store_mip_image);
    const auto cached_zero_store_mip = recompile_compute_shader_cached(
        kZeroStoreMipCompute, std::size(kZeroStoreMipCompute),
        &zero_store_mip_table, zero_mip_config);
    zero_store_mip_table.resources[0].declared_mip_levels = 2;
    const auto cached_multilevel_store_mip = recompile_compute_shader_cached(
        kZeroStoreMipCompute, std::size(kZeroStoreMipCompute),
        &zero_store_mip_table, zero_mip_config);
    stats = shader_recompile_cache_stats();
    CHECK(!cached_zero_store_mip.empty() && cached_multilevel_store_mip.empty() &&
              stats.misses == 2 && stats.hits == 0 && stats.entries == 2,
          "compute cache keys storage-image declared mip levels inspected by the emitter");

    // A one-record 16-bit V# emits an explicit index guard and typed tail marker. Keep the cache
    // partition compact (none/u16/f16), while ensuring a wider range or the sibling conversion can
    // never reuse that specialized module.
    clear_shader_recompile_cache();
    static const uint32_t kTailCompute[] = {
        0xe0002000u, 0x80020100u, // buffer_load_format_x v1, v0, s[8:11], idxen
        0xbf810000u,
    };
    ShaderResource tail_buffer;
    tail_buffer.cls = ResourceClass::ConstantBuffer;
    tail_buffer.format = DataFormat::Uint16;
    tail_buffer.num_components = 1;
    tail_buffer.binding = 3;
    tail_buffer.size = 2;
    tail_buffer.stride = 2;
    tail_buffer.sgpr_base = 8;
    tail_buffer.fetch_pc = 0;
    ShaderResourceTable tail_table;
    tail_table.resources.push_back(tail_buffer);
    ComputeShaderConfig tail_config;
    tail_config.user_sgprs.resize(12);
    tail_config.local_x = 64;
    const auto cached_u16_tail = recompile_compute_shader_cached(
        kTailCompute, std::size(kTailCompute), &tail_table, tail_config);
    tail_table.resources[0].size = 4;
    const auto cached_wide_tail = recompile_compute_shader_cached(
        kTailCompute, std::size(kTailCompute), &tail_table, tail_config);
    tail_table.resources[0].size = 2;
    tail_table.resources[0].format = DataFormat::Float16;
    const auto cached_f16_tail = recompile_compute_shader_cached(
        kTailCompute, std::size(kTailCompute), &tail_table, tail_config);
    tail_table.resources[0].format = DataFormat::Uint16;
    const auto repeated_u16_tail = recompile_compute_shader_cached(
        kTailCompute, std::size(kTailCompute), &tail_table, tail_config);
    stats = shader_recompile_cache_stats();
    CHECK(!cached_u16_tail.empty() && !cached_wide_tail.empty() && !cached_f16_tail.empty() &&
              cached_u16_tail != cached_wide_tail && cached_u16_tail != cached_f16_tail &&
              repeated_u16_tail == cached_u16_tail && stats.misses == 3 && stats.hits == 1,
          "compute cache partitions ordinary/u16/f16 one-record tail semantics exactly");

    // A zero-record RAW V# is compiled into constants/no-ops, while a later nonzero descriptor at
    // the same instruction emits a real buffer access. Addresses and ordinary sizes intentionally
    // stay out of the module key, so partition only this exact semantic marker.
    clear_shader_recompile_cache();
    static const uint32_t kZeroRecordRawCompute[] = {
        0xe0300000u, 0x80000000u, // buffer_load_dword v0, off, s[0:3]
        0xbf810000u,
    };
    ShaderResource zero_record_raw;
    zero_record_raw.cls = ResourceClass::ConstantBuffer;
    zero_record_raw.format = DataFormat::Unknown;
    zero_record_raw.num_components = 0;
    zero_record_raw.binding = 3;
    zero_record_raw.gpu_addr = 0;
    zero_record_raw.size = 0;
    zero_record_raw.stride = 0;
    zero_record_raw.srt_offset = 0xFFFFFFFFu;
    zero_record_raw.sgpr_base = 0xFFFFFFFFu;
    zero_record_raw.fetch_pc = 0;
    ShaderResourceTable zero_record_raw_table;
    zero_record_raw_table.resources.push_back(zero_record_raw);
    ComputeShaderConfig zero_record_raw_config;
    zero_record_raw_config.user_sgprs.resize(4);
    zero_record_raw_config.local_x = 64;
    const auto cached_zero_record_raw = recompile_compute_shader_cached(
        kZeroRecordRawCompute, std::size(kZeroRecordRawCompute), &zero_record_raw_table,
        zero_record_raw_config);
    zero_record_raw_table.resources[0].gpu_addr = 0x20000u;
    zero_record_raw_table.resources[0].size = 4;
    const auto cached_nonzero_raw = recompile_compute_shader_cached(
        kZeroRecordRawCompute, std::size(kZeroRecordRawCompute), &zero_record_raw_table,
        zero_record_raw_config);
    zero_record_raw_table.resources[0].gpu_addr = 0;
    zero_record_raw_table.resources[0].size = 0;
    const auto repeated_zero_record_raw = recompile_compute_shader_cached(
        kZeroRecordRawCompute, std::size(kZeroRecordRawCompute), &zero_record_raw_table,
        zero_record_raw_config);
    stats = shader_recompile_cache_stats();
    CHECK(is_zero_record_raw_buffer(zero_record_raw_table.resources[0]) &&
              !cached_zero_record_raw.empty() && !cached_nonzero_raw.empty() &&
              cached_zero_record_raw != cached_nonzero_raw &&
              repeated_zero_record_raw == cached_zero_record_raw && stats.misses == 2 &&
              stats.hits == 1 && stats.entries == 2,
          "compute cache separates exact zero-record RAW lowering from nonzero buffers");

    // The application-level optional-null marker has different load-only semantics from an
    // ordinary explicit null resource. Its serialized sampler provenance sentinel is inert for a
    // ConstantBuffer, so key the semantic predicate explicitly rather than its unused raw field.
    clear_shader_recompile_cache();
    static const uint32_t kOptionalNullRawCompute[] = {
        0xe0302000u, 0x80000000u, // buffer_load_dword v0, v0, s[0:3], idxen
        0xbf810000u,
    };
    ShaderResource optional_null_raw;
    optional_null_raw.cls = ResourceClass::ConstantBuffer;
    optional_null_raw.format = DataFormat::Uint32;
    optional_null_raw.num_components = 1;
    optional_null_raw.binding = 3;
    optional_null_raw.stride = kGtaOptionalBufferStride;
    optional_null_raw.fetch_pc = 0;
    optional_null_raw.sampler_sgpr_base = kOptionalNullRawLoadMarkerSamplerBase;
    ShaderResourceTable optional_null_raw_table;
    optional_null_raw_table.resources.push_back(optional_null_raw);
    ComputeShaderConfig optional_null_raw_config;
    optional_null_raw_config.user_sgprs.resize(4);
    optional_null_raw_config.local_x = kGtaOptionalBufferLocalSize;
    const auto cached_optional_null_raw = recompile_compute_shader_cached(
        kOptionalNullRawCompute, std::size(kOptionalNullRawCompute),
        &optional_null_raw_table, optional_null_raw_config);
    optional_null_raw_table.resources[0].sampler_sgpr_base = 0xFFFFFFFFu;
    const auto cached_ordinary_null_raw = recompile_compute_shader_cached(
        kOptionalNullRawCompute, std::size(kOptionalNullRawCompute),
        &optional_null_raw_table, optional_null_raw_config);
    optional_null_raw_table.resources[0].sampler_sgpr_base =
        kOptionalNullRawLoadMarkerSamplerBase;
    const auto repeated_optional_null_raw = recompile_compute_shader_cached(
        kOptionalNullRawCompute, std::size(kOptionalNullRawCompute),
        &optional_null_raw_table, optional_null_raw_config);
    stats = shader_recompile_cache_stats();
    CHECK(is_optional_null_raw_load_buffer(optional_null_raw_table.resources[0]) &&
              !cached_optional_null_raw.empty() &&
              cached_optional_null_raw != cached_ordinary_null_raw &&
              repeated_optional_null_raw == cached_optional_null_raw &&
              stats.misses == 2 && stats.hits == 1 && stats.entries == 2,
          "compute cache separates optional-null load lowering from ordinary null resources");
    // Guarded-null stores compile to no memory operation too, but for a different reason: their V#
    // has one record and only the exact dispatch CFG proves the store unreachable. Partition that
    // marker explicitly from an ordinary writable resource at the same fetch pc.
    clear_shader_recompile_cache();
    std::array<uint32_t, 81> guarded_null_store_compute;
    guarded_null_store_compute.fill(0xbf800000u); // s_nop 0
    guarded_null_store_compute[22] = 0xbe92047eu; // s_mov_b64 s[18:19], exec
    guarded_null_store_compute[41] = 0xbefe0412u; // s_mov_b64 exec, s[18:19]
    guarded_null_store_compute[42] = 0xbf128002u;
    guarded_null_store_compute[43] = 0x7d8a00f9u;
    guarded_null_store_compute[44] = 0x06868080u;
    guarded_null_store_compute[45] = 0x85ea8012u;
    guarded_null_store_compute[46] = 0x8dea006au;
    guarded_null_store_compute[47] = 0x87fe126au;
    guarded_null_store_compute[48] = 0xbf88001fu;
    guarded_null_store_compute[74] = 0xe0740030u;
    guarded_null_store_compute[75] = 0x80000700u;
    guarded_null_store_compute[80] = 0xbf810000u;
    ShaderResource guarded_null_store = zero_record_raw;
    guarded_null_store.fetch_pc = 74u;
    guarded_null_store.stride = kProvenNullGuardedRawStoreStride;
    ShaderResourceTable guarded_null_store_table;
    guarded_null_store_table.resources.push_back(guarded_null_store);
    ComputeShaderConfig stale_guarded_store_config = zero_record_raw_config;
    stale_guarded_store_config.user_sgprs[2] = 1u;
    // Exercise both sides of the cache boundary: invalid conditional metadata must fail before a
    // cold lookup and must still fail after the valid null-dispatch module has warmed the cache.
    const auto cold_stale_guarded_null_store = recompile_compute_shader_cached(
        guarded_null_store_compute.data(), guarded_null_store_compute.size(),
        &guarded_null_store_table, stale_guarded_store_config);
    const auto cached_guarded_null_store = recompile_compute_shader_cached(
        guarded_null_store_compute.data(), guarded_null_store_compute.size(),
        &guarded_null_store_table, zero_record_raw_config);
    const auto warm_stale_guarded_null_store = recompile_compute_shader_cached(
        guarded_null_store_compute.data(), guarded_null_store_compute.size(),
        &guarded_null_store_table, stale_guarded_store_config);
    std::array<uint32_t, 81> overwritten_guarded_null_store_compute =
        guarded_null_store_compute;
    overwritten_guarded_null_store_compute[40] = 0xbe820381u; // s_mov_b32 s2, 1
    const auto overwritten_guarded_null_store = recompile_compute_shader_cached(
        overwritten_guarded_null_store_compute.data(),
        overwritten_guarded_null_store_compute.size(),
        &guarded_null_store_table, zero_record_raw_config);
    guarded_null_store_table.resources[0].gpu_addr = 0x20000u;
    guarded_null_store_table.resources[0].size = 4u;
    guarded_null_store_table.resources[0].stride = 4u;
    const auto cached_ordinary_store = recompile_compute_shader_cached(
        guarded_null_store_compute.data(), guarded_null_store_compute.size(),
        &guarded_null_store_table, zero_record_raw_config);
    guarded_null_store_table.resources[0] = guarded_null_store;
    const auto repeated_guarded_null_store = recompile_compute_shader_cached(
        guarded_null_store_compute.data(), guarded_null_store_compute.size(),
        &guarded_null_store_table, zero_record_raw_config);
    stats = shader_recompile_cache_stats();
    CHECK(is_proven_null_guarded_raw_store(guarded_null_store_table.resources[0]) &&
              cold_stale_guarded_null_store.empty() &&
              warm_stale_guarded_null_store.empty() &&
              overwritten_guarded_null_store.empty() &&
              !cached_guarded_null_store.empty() && !cached_ordinary_store.empty() &&
              cached_guarded_null_store != cached_ordinary_store &&
              repeated_guarded_null_store == cached_guarded_null_store &&
              stats.misses == 2u && stats.hits == 1u && stats.entries == 2u,
          "compute cache separates guarded-null store elision from ordinary writable buffers");

    // GTA V's exact workgroup-store kernel retains the mapped dispatch-table bytes that proved its
    // optional +0x20 output pointer null. That witness is dispatch state rather than compile-key
    // state, so it must be revalidated before a warm lookup can reuse the store-eliding module.
    clear_shader_recompile_cache();
    alignas(16) std::array<uint32_t, kGtaNullableOutputWitnessBytes / sizeof(uint32_t)>
        nullable_output_witness{};
    std::array<uint32_t, 64u * 4u> nullable_input_a{};
    std::array<uint32_t, 64u * 4u> nullable_input_b{};
    auto split_address = [](uint64_t address, uint32_t& low, uint32_t& high) {
        low = static_cast<uint32_t>(address);
        high = static_cast<uint32_t>(address >> 32u);
    };
    std::array<uint32_t, 9> nullable_output_seed{};
    split_address(reinterpret_cast<uint64_t>(nullable_output_witness.data()),
                  nullable_output_seed[0], nullable_output_seed[1]);
    split_address(reinterpret_cast<uint64_t>(nullable_input_a.data()),
                  nullable_output_seed[2], nullable_output_seed[3]);
    split_address(reinterpret_cast<uint64_t>(nullable_input_b.data()),
                  nullable_output_seed[4], nullable_output_seed[5]);
    nullable_output_seed[6] = 64u;
    nullable_output_seed[7] = kGtaNullableOutputFixtureRecordCount;
    nullable_output_seed[8] = kGtaNullableOutputUserSgpr8;

    ComputeResourceDispatchContext nullable_output_context;
    nullable_output_context.local_x = kGtaNullableOutputLocalSize;
    nullable_output_context.local_y = nullable_output_context.local_z = 1u;
    nullable_output_context.threads_x = kGtaNullableOutputFixtureThreads;
    nullable_output_context.threads_y = nullable_output_context.threads_z = 1u;
    nullable_output_context.wave_size = 64u;
    nullable_output_context.tgid_x_en = true;
    nullable_output_context.tidig_comp_cnt = 0u;
    ShaderResourceTable nullable_output_table;
    (void)add_compute_buffer_resources(
        nullable_output_table, prosper::test::kGta5WorkgroupStoreProgram.data(),
        prosper::test::kGta5WorkgroupStoreProgram.size(), nullable_output_seed.data(),
        nullable_output_seed.size(), nullable_output_context.local_x,
        nullable_output_context.threads_x, static_cast<uint32_t>(nullable_output_seed.size()),
        &nullable_output_context);
    assign_convention_bindings(nullable_output_table, 2u);
    for (ShaderResource& resource : nullable_output_table.resources) {
        if (!is_proven_null_nullable_raw_buffer(resource)) continue;
        resource.host_data = reinterpret_cast<uint8_t*>(nullable_output_witness.data());
        resource.host_data_size = kGtaNullableOutputWitnessBytes;
    }

    ComputeShaderConfig nullable_output_config;
    nullable_output_config.user_sgprs.assign(nullable_output_seed.begin(),
                                              nullable_output_seed.end());
    nullable_output_config.local_x = nullable_output_context.local_x;
    nullable_output_config.local_y = nullable_output_context.local_y;
    nullable_output_config.local_z = nullable_output_context.local_z;
    nullable_output_config.threads_x = nullable_output_context.threads_x;
    nullable_output_config.threads_y = nullable_output_context.threads_y;
    nullable_output_config.threads_z = nullable_output_context.threads_z;
    nullable_output_config.wave_size = nullable_output_context.wave_size;
    nullable_output_config.tgid_x_en = nullable_output_context.tgid_x_en;
    uint64_t nullable_cold_identity = 0u;
    const auto nullable_cold = recompile_compute_shader_cached(
        prosper::test::kGta5WorkgroupStoreProgram.data(),
        prosper::test::kGta5WorkgroupStoreProgram.size(), &nullable_output_table,
        nullable_output_config, &nullable_cold_identity);

    // Mutate the same retained producer bytes consumed by the production pre-cache validator. A
    // different helper/marker mutation would not prove the warm-cache trust boundary itself.
    nullable_output_witness[kGtaNullableOutputPointerOffset / sizeof(uint32_t)] = 1u;
    uint64_t nullable_stale_identity = 1u;
    const auto nullable_stale = recompile_compute_shader_cached(
        prosper::test::kGta5WorkgroupStoreProgram.data(),
        prosper::test::kGta5WorkgroupStoreProgram.size(), &nullable_output_table,
        nullable_output_config, &nullable_stale_identity);
    nullable_output_witness[kGtaNullableOutputPointerOffset / sizeof(uint32_t)] = 0u;
    uint64_t nullable_warm_identity = 0u;
    const auto nullable_warm = recompile_compute_shader_cached(
        prosper::test::kGta5WorkgroupStoreProgram.data(),
        prosper::test::kGta5WorkgroupStoreProgram.size(), &nullable_output_table,
        nullable_output_config, &nullable_warm_identity);
    stats = shader_recompile_cache_stats();
    CHECK(std::count_if(nullable_output_table.resources.begin(),
                        nullable_output_table.resources.end(),
                        is_proven_null_nullable_raw_buffer) == 1u &&
              !nullable_cold.empty() && nullable_stale.empty() &&
              nullable_stale_identity == 0u && nullable_warm == nullable_cold &&
              nullable_cold_identity != 0u && nullable_warm_identity == nullable_cold_identity &&
              stats.misses == 1u && stats.hits == 1u && stats.entries == 1u,
          "nullable-output cache revalidates its retained +0x20 witness before warm reuse");

    // Resource descriptor fields that specialize SPIR-V must participate in the cache identity.
    // Storage-image sRGB state changes whether the module declares a native float image or the raw
    // uint-channel fallback, even when the guest code and every binding/provenance field are equal.
    clear_shader_recompile_cache();
    static const uint32_t kStorageCompute[] = {
        0x7e080300u, 0x7e0a0280u,
        0xf0000f08u, 0x00000004u, 0xbf8c3f70u,
        0xf0200f08u, 0x00020004u, 0xbf810000u,
    };
    ShaderResourceTable storage_table;
    for (uint32_t i = 0; i < 2; ++i) {
        ShaderResource image;
        image.cls = ResourceClass::StorageImage;
        image.format = DataFormat::Float32;
        image.num_components = 4;
        image.binding = 4 + i;
        image.sgpr_base = i * 8;
        image.img_dim = 1;
        image.width = image.height = image.depth = 1;
        storage_table.resources.push_back(image);
    }
    ComputeShaderConfig storage_config;
    storage_config.user_sgprs.resize(16);
    storage_config.local_x = 64;
    storage_config.native_storage_format_support =
        native_storage_format_support_bit(DataFormat::Float32, 4);
    uint64_t native_storage_identity = 0;
    const auto native_storage = recompile_compute_shader_cached(
        kStorageCompute, std::size(kStorageCompute), &storage_table, storage_config,
        &native_storage_identity);
    storage_config.native_storage_format_support = 0;
    uint64_t raw_storage_identity = 0;
    const auto raw_storage = recompile_compute_shader_cached(
        kStorageCompute, std::size(kStorageCompute), &storage_table, storage_config,
        &raw_storage_identity);
    storage_config.native_storage_format_support =
        native_storage_format_support_bit(DataFormat::Float32, 4);
    storage_table.resources[0].srgb = true;
    uint64_t srgb_storage_identity = 0;
    const auto srgb_storage = recompile_compute_shader_cached(
        kStorageCompute, std::size(kStorageCompute), &storage_table, storage_config,
        &srgb_storage_identity);
    storage_table.resources[0].srgb = false;
    uint64_t repeated_storage_identity = 0;
    const auto repeated_storage = recompile_compute_shader_cached(
        kStorageCompute, std::size(kStorageCompute), &storage_table, storage_config,
        &repeated_storage_identity);
    stats = shader_recompile_cache_stats();
    const auto native_report = validate_spirv_descriptor_interface(
        native_storage, &storage_table, 0, SpirvShaderStage::Compute, false);
    const auto raw_report = validate_spirv_descriptor_interface(
        raw_storage, &storage_table, 0, SpirvShaderStage::Compute, false);
    CHECK(!native_storage.empty() && !raw_storage.empty() && !srgb_storage.empty() &&
              native_storage != raw_storage && native_storage != srgb_storage &&
              repeated_storage == native_storage &&
              native_storage_identity != 0 && raw_storage_identity != 0 &&
              srgb_storage_identity != 0 &&
              native_storage_identity != srgb_storage_identity &&
              native_storage_identity != raw_storage_identity &&
              repeated_storage_identity == native_storage_identity &&
              native_report.descriptors.size() == 2 &&
              native_report.descriptors[0].storage_float &&
              raw_report.descriptors.size() == 2 &&
              !raw_report.descriptors[0].storage_float &&
              stats.misses == 3 && stats.hits == 1 && stats.entries == 3,
          "compute cache separates device-gated typed storage from the raw fallback and sRGB state");

    // The exact R11G11B10 path is an explicit A/B contract: enabled modules use one typed R32ui
    // word per texel even when native typed storage is available. The recovery switch still
    // exposes the device-gated typed-float path, making both the specialization and cache lever
    // independently observable.
    clear_shader_recompile_cache();
    static const uint32_t kPackedStorageCompute[] = {
        0x7e080300u, 0x7e0a0280u,
        0xf0000f08u, 0x00000004u, 0xbf8c3f70u,
        0xf0200f08u, 0x00020004u, 0xbf810000u,
    };
    ShaderResourceTable packed_storage_table;
    for (uint32_t i = 0; i < 2; ++i) {
        ShaderResource image;
        image.cls = ResourceClass::StorageImage;
        image.format = DataFormat::Float10_11_11;
        image.num_components = 3;
        image.img_dim = 1;
        image.width = 64;
        image.height = image.depth = 1;
        image.size = 64 * sizeof(uint32_t);
        image.binding = 4 + i;
        image.sgpr_base = i * 8;
        packed_storage_table.resources.push_back(image);
    }
    ComputeShaderConfig packed_storage_config;
    packed_storage_config.user_sgprs.resize(16);
    packed_storage_config.local_x = 64;
    packed_storage_config.native_storage_format_support =
        native_storage_format_support_bit(DataFormat::Float10_11_11, 3);
    uint64_t packed_storage_identity = 0;
    const auto packed_storage = recompile_compute_shader_cached(
        kPackedStorageCompute, std::size(kPackedStorageCompute), &packed_storage_table,
        packed_storage_config, &packed_storage_identity);
    packed_storage_config.packed_r11_storage = false;
    uint64_t converted_storage_identity = 0;
    const auto converted_storage = recompile_compute_shader_cached(
        kPackedStorageCompute, std::size(kPackedStorageCompute), &packed_storage_table,
        packed_storage_config, &converted_storage_identity);
    packed_storage_config.packed_r11_storage = true;
    uint64_t repeated_packed_storage_identity = 0;
    const auto repeated_packed_storage = recompile_compute_shader_cached(
        kPackedStorageCompute, std::size(kPackedStorageCompute), &packed_storage_table,
        packed_storage_config, &repeated_packed_storage_identity);
    const auto packed_storage_report = validate_spirv_descriptor_interface(
        packed_storage, &packed_storage_table, 0, SpirvShaderStage::Compute, false);
    const auto converted_storage_report = validate_spirv_descriptor_interface(
        converted_storage, &packed_storage_table, 0, SpirvShaderStage::Compute, false);
    stats = shader_recompile_cache_stats();
    CHECK(!packed_storage.empty() && !converted_storage.empty() &&
              packed_storage != converted_storage && repeated_packed_storage == packed_storage &&
              packed_storage_identity != 0 && converted_storage_identity != 0 &&
              packed_storage_identity != converted_storage_identity &&
              repeated_packed_storage_identity == packed_storage_identity &&
              packed_storage_report.descriptors.size() == 2 &&
              !packed_storage_report.descriptors[0].storage_float &&
              packed_storage_report.descriptors[0].storage_image_format == kSpirvImageFormatR32ui &&
              converted_storage_report.descriptors.size() == 2 &&
              converted_storage_report.descriptors[0].storage_float &&
              converted_storage_report.descriptors[0].storage_image_format == 0 &&
              stats.misses == 2 && stats.hits == 1 && stats.entries == 2,
          "compute cache keeps exact packed R11 authoritative over native typed storage");

    // Compute image atomics embed the runtime R32_UINT extent in the linearized SSBO index and
    // bounds guard. Reusing a module across extents would retain a stale row stride or bounds.
    clear_shader_recompile_cache();
    static const uint32_t kImageAtomicCompute[] = {
        0x7e000280u, 0x7e020280u, 0x7e120281u,
        0xf0442108u, 0x00000900u, 0xbf810000u,
    };
    uint32_t atomic_pixels[32] = {};
    ShaderResource atomic_image;
    atomic_image.cls = ResourceClass::StorageImage;
    atomic_image.format = DataFormat::Uint32;
    atomic_image.num_components = 1;
    atomic_image.binding = 4;
    atomic_image.sgpr_base = 0;
    atomic_image.img_dim = 1;
    atomic_image.width = 4;
    atomic_image.height = 2;
    atomic_image.depth = 1;
    atomic_image.size = sizeof(atomic_pixels);
    atomic_image.host_data = reinterpret_cast<uint8_t*>(atomic_pixels);
    atomic_image.host_data_size = sizeof(atomic_pixels);
    ShaderResourceTable atomic_table;
    atomic_table.resources.push_back(atomic_image);
    ComputeShaderConfig atomic_config;
    atomic_config.user_sgprs.resize(8);
    atomic_config.local_x = 1;
    const auto atomic_4x2 = recompile_compute_shader_cached(
        kImageAtomicCompute, std::size(kImageAtomicCompute), &atomic_table, atomic_config);
    atomic_table.resources[0].width = 8;
    const auto atomic_8x2 = recompile_compute_shader_cached(
        kImageAtomicCompute, std::size(kImageAtomicCompute), &atomic_table, atomic_config);
    atomic_table.resources[0].width = 4;
    const auto atomic_4x2_again = recompile_compute_shader_cached(
        kImageAtomicCompute, std::size(kImageAtomicCompute), &atomic_table, atomic_config);
    stats = shader_recompile_cache_stats();
    CHECK(!atomic_4x2.empty() && !atomic_8x2.empty() &&
              atomic_4x2 != atomic_8x2 && atomic_4x2_again == atomic_4x2 &&
              stats.misses == 2 && stats.hits == 1 && stats.entries == 2,
          "compute cache separates image-atomic modules with different embedded extents");

    // Qword-atomic code generation depends on the enabled device feature, exact linear descriptor
    // proof, and dispatch-sized record count embedded in the OOB check. All three must partition the
    // cache so one dispatch cannot lend its module or bound to a different/unproved sibling.
    clear_shader_recompile_cache();
    static const uint32_t kSwapX2Compute[] = {
        0xe1402000u, 0x80000900u,
        0xbf810000u,
    };
    alignas(8) uint32_t swap_x2_words[50] = {};
    ShaderResource swap_x2_resource;
    swap_x2_resource.cls = ResourceClass::ConstantBuffer;
    swap_x2_resource.format = DataFormat::Uint32;
    swap_x2_resource.num_components = 1;
    swap_x2_resource.binding = 2;
    swap_x2_resource.gpu_addr = reinterpret_cast<uint64_t>(swap_x2_words);
    swap_x2_resource.size = 200;
    swap_x2_resource.stride = 8;
    swap_x2_resource.sgpr_base = 0;
    swap_x2_resource.fetch_pc = 0;
    swap_x2_resource.atomic_x2_record_count = 25;
    ShaderResourceTable swap_x2_table;
    swap_x2_table.resources.push_back(swap_x2_resource);
    ComputeShaderConfig swap_x2_config;
    swap_x2_config.local_x = 1;
    swap_x2_config.storage_buffer_int64_atomics = true;
    const auto swap_x2 = recompile_compute_shader_cached(
        kSwapX2Compute, std::size(kSwapX2Compute), &swap_x2_table, swap_x2_config);
    ShaderResourceTable two_record_swap_x2_table = swap_x2_table;
    two_record_swap_x2_table.resources[0].size = 16;
    two_record_swap_x2_table.resources[0].atomic_x2_record_count = 2;
    const auto two_record_swap_x2 = recompile_compute_shader_cached(
        kSwapX2Compute, std::size(kSwapX2Compute),
        &two_record_swap_x2_table, swap_x2_config);
    ShaderResourceTable oversized_swap_x2_table = swap_x2_table;
    oversized_swap_x2_table.resources[0].size = 0x10000008u;
    oversized_swap_x2_table.resources[0].atomic_x2_record_count = 0x02000001u;
    const auto oversized_swap_x2 = recompile_compute_shader_cached(
        kSwapX2Compute, std::size(kSwapX2Compute),
        &oversized_swap_x2_table, swap_x2_config);
    // These retain the opaque marker but violate separate resource-side emitter gates. They run
    // immediately after the valid warm entry: if the cache keys only the marker, each silently
    // borrows that module without reaching the gate it mutated.
    ShaderResourceTable wrong_size_swap_x2_table = swap_x2_table;
    wrong_size_swap_x2_table.resources[0].size = 198;
    const auto wrong_size_swap_x2 = recompile_compute_shader_cached(
        kSwapX2Compute, std::size(kSwapX2Compute),
        &wrong_size_swap_x2_table, swap_x2_config);
    ShaderResourceTable misaligned_swap_x2_table = swap_x2_table;
    misaligned_swap_x2_table.resources[0].gpu_addr += 4;
    const auto misaligned_swap_x2 = recompile_compute_shader_cached(
        kSwapX2Compute, std::size(kSwapX2Compute),
        &misaligned_swap_x2_table, swap_x2_config);
    ShaderResourceTable indexed_swap_x2_table = swap_x2_table;
    indexed_swap_x2_table.resources[0].table_index_count = 1;
    const auto indexed_swap_x2 = recompile_compute_shader_cached(
        kSwapX2Compute, std::size(kSwapX2Compute),
        &indexed_swap_x2_table, swap_x2_config);
    ComputeShaderConfig no_int64_atomic = swap_x2_config;
    no_int64_atomic.storage_buffer_int64_atomics = false;
    const auto unsupported_swap_x2 = recompile_compute_shader_cached(
        kSwapX2Compute, std::size(kSwapX2Compute), &swap_x2_table, no_int64_atomic);
    ShaderResourceTable unproved_swap_x2_table = swap_x2_table;
    unproved_swap_x2_table.resources[0].atomic_x2_record_count = 0;
    const auto unproved_swap_x2 = recompile_compute_shader_cached(
        kSwapX2Compute, std::size(kSwapX2Compute), &unproved_swap_x2_table, swap_x2_config);
    const auto swap_x2_again = recompile_compute_shader_cached(
        kSwapX2Compute, std::size(kSwapX2Compute), &swap_x2_table, swap_x2_config);
    const auto two_record_swap_x2_again = recompile_compute_shader_cached(
        kSwapX2Compute, std::size(kSwapX2Compute),
        &two_record_swap_x2_table, swap_x2_config);
    stats = shader_recompile_cache_stats();
    CHECK(!swap_x2.empty() && !two_record_swap_x2.empty() && oversized_swap_x2.empty() &&
              swap_x2 != two_record_swap_x2 && wrong_size_swap_x2.empty() &&
              misaligned_swap_x2.empty() && indexed_swap_x2.empty() &&
              unsupported_swap_x2.empty() && unproved_swap_x2.empty() &&
              swap_x2_again == swap_x2 && two_record_swap_x2_again == two_record_swap_x2 &&
              stats.misses == 5 && stats.hits == 5 && stats.entries == 5,
          "swap_x2 cache separates record count, capability, and every admission gate");

    // BVH BOX_GROW is embedded as the slab-test far-edge multiplier. A descriptor change must
    // compile a distinct module even when the BVH allocation and instruction provenance match.
    clear_shader_recompile_cache();
    static const uint32_t kBvhCompute[] = {
        0xf1989f07u, 0x00040303u, 0x43440d3fu, 0x46424140u, 0x00004847u,
        0xbf810000u,
    };
    ShaderResource bvh;
    bvh.cls = ResourceClass::ConstantBuffer;
    bvh.format = DataFormat::Uint32;
    bvh.num_components = 1;
    bvh.binding = 4;
    bvh.size = 128;
    bvh.fetch_pc = 0;
    ShaderResourceTable bvh_table;
    bvh_table.resources.push_back(bvh);
    ComputeShaderConfig bvh_config;
    bvh_config.local_x = 1;
    const auto bvh_grow_0 = recompile_compute_shader_cached(
        kBvhCompute, std::size(kBvhCompute), &bvh_table, bvh_config);
    bvh_table.resources[0].bvh_box_grow = 6;
    const auto bvh_grow_6 = recompile_compute_shader_cached(
        kBvhCompute, std::size(kBvhCompute), &bvh_table, bvh_config);
    bvh_table.resources[0].bvh_box_grow = 0;
    const auto bvh_grow_0_again = recompile_compute_shader_cached(
        kBvhCompute, std::size(kBvhCompute), &bvh_table, bvh_config);
    bvh_table.resources[0].bvh_sort_enabled = true;
    const auto sorted_bvh = recompile_compute_shader_cached(
        kBvhCompute, std::size(kBvhCompute), &bvh_table, bvh_config);
    bvh_table.resources[0].bvh_sort_enabled = false;
    alignas(256) static uint32_t null_bvh_words[64] = {};
    bvh_table.resources[0].size = sizeof(null_bvh_words);
    bvh_table.resources[0].host_data = reinterpret_cast<uint8_t*>(null_bvh_words);
    bvh_table.resources[0].host_data_size = sizeof(null_bvh_words);
    const auto guarded_null_bvh = recompile_compute_shader_cached(
        kBvhCompute, std::size(kBvhCompute), &bvh_table, bvh_config);
    bvh_table.resources[0].size = 128;
    bvh_table.resources[0].host_data = nullptr;
    bvh_table.resources[0].host_data_size = 0;
    const auto bvh_grow_0_after_null = recompile_compute_shader_cached(
        kBvhCompute, std::size(kBvhCompute), &bvh_table, bvh_config);
    stats = shader_recompile_cache_stats();
    CHECK(!bvh_grow_0.empty() && !bvh_grow_6.empty() &&
              bvh_grow_0 != bvh_grow_6 && bvh_grow_0_again == bvh_grow_0 &&
              !sorted_bvh.empty() && sorted_bvh != bvh_grow_0 &&
              !guarded_null_bvh.empty() && guarded_null_bvh != bvh_grow_0 &&
              bvh_grow_0_after_null == bvh_grow_0 &&
              stats.misses == 4 && stats.hits == 2 && stats.entries == 4,
          "compute cache separates BVH box growth, sorting, and guarded-null lowering");

    // Manual shadow comparison bakes the enable, compare op, filter mode, address modes, and border
    // color into SPIR-V. In particular depth_compare=false must not reuse a previously successful
    // module: that descriptor contract is supposed to reject this comparison-sample shader.
    clear_shader_recompile_cache();
    static const uint32_t kShadowPs[] = {
        0xc8080000u, 0xc8090001u, 0xc80c0100u, 0xc80d0101u,
        0x7e0202f0u, 0xf0bc0108u, 0x00820401u,
        0xf800080fu, 0x07060504u, 0xbf810000u,
    };
    ShaderResource shadow;
    shadow.cls = ResourceClass::Texture;
    shadow.format = DataFormat::Unorm8;
    shadow.num_components = 4;
    shadow.binding = 4;
    shadow.sgpr_base = 8;
    shadow.depth_compare = true;
    shadow.depth_compare_func = 4;
    shadow.mag_filter = 0;
    ShaderResourceTable shadow_table;
    shadow_table.resources.push_back(shadow);
    uint64_t shadow_identity = 0, shadow_repeat_identity = 0, disabled_identity = 0;
    uint64_t function_identity = 0, filter_identity = 0, address_identity = 0, border_identity = 0;
    const auto shadow_spv = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kShadowPs, std::size(kShadowPs), &shadow_table,
        nullptr, nullptr, &shadow_identity);
    const auto shadow_repeat = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kShadowPs, std::size(kShadowPs), &shadow_table,
        nullptr, nullptr, &shadow_repeat_identity);
    shadow_table.resources[0].depth_compare = false;
    const auto disabled_shadow = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kShadowPs, std::size(kShadowPs), &shadow_table,
        nullptr, nullptr, &disabled_identity);
    shadow_table.resources[0].depth_compare = true;
    shadow_table.resources[0].depth_compare_func = 1;
    const auto function_shadow = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kShadowPs, std::size(kShadowPs), &shadow_table,
        nullptr, nullptr, &function_identity);
    shadow_table.resources[0].mag_filter = 1;
    const auto filter_shadow = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kShadowPs, std::size(kShadowPs), &shadow_table,
        nullptr, nullptr, &filter_identity);
    shadow_table.resources[0].addr_uvw[0] = 6;
    const auto address_shadow = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kShadowPs, std::size(kShadowPs), &shadow_table,
        nullptr, nullptr, &address_identity);
    shadow_table.resources[0].border_color_type = 2;
    const auto border_shadow = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kShadowPs, std::size(kShadowPs), &shadow_table,
        nullptr, nullptr, &border_identity);
    stats = shader_recompile_cache_stats();
    CHECK(!shadow_spv.empty() && shadow_repeat == shadow_spv && disabled_shadow.empty() &&
              !function_shadow.empty() && !filter_shadow.empty() && !address_shadow.empty() &&
              !border_shadow.empty() && shadow_identity == shadow_repeat_identity &&
              shadow_identity != function_identity && function_identity != filter_identity &&
              filter_identity != address_identity && address_identity != border_identity &&
              stats.misses == 6 && stats.hits == 1 && stats.entries == 6,
          "shader cache separates every manual shadow-comparison codegen field");

    // The exact Asterix NSA 2D_MSAA IMAGE_LOAD lowers four guest samples to four host array layers.
    // sample_count is therefore codegen state, not upload-only descriptor state: a cached 4x module
    // must never satisfy the deliberately unsupported 2x table, while an identical 4x table must hit.
    clear_shader_recompile_cache();
    static const uint32_t kMsaaLoadPs[] = {
        0x7e0a0280u,                       // v_mov_b32 v5, 0 (x)
        0x7e0c0280u,                       // v_mov_b32 v6, 0 (y)
        0x7e040281u,                       // v_mov_b32 v2, 1 (sample)
        0xf0000132u, 0x00000205u, 0x00000206u, // image_load v2, [v5,v6,v2], s[0:7]
        0xbf8c0f70u,                       // s_waitcnt vmcnt(0)
        0xf800180fu, 0x02020202u,         // exp mrt0 v2,v2,v2,v2
        0xbf810000u,
    };
    ShaderResource msaa_texture;
    msaa_texture.cls = ResourceClass::Texture;
    msaa_texture.format = DataFormat::Float32;
    msaa_texture.num_components = 1;
    msaa_texture.binding = 4;
    msaa_texture.sgpr_base = 0;
    msaa_texture.img_dim = 6;
    msaa_texture.width = 1;
    msaa_texture.height = 1;
    msaa_texture.depth = 1;
    msaa_texture.sample_count = 4;
    msaa_texture.declared_mip_levels = 1;
    ShaderResourceTable msaa_table;
    msaa_table.resources.push_back(msaa_texture);
    const auto msaa_4x = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kMsaaLoadPs, std::size(kMsaaLoadPs), &msaa_table);
    const auto msaa_4x_again = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kMsaaLoadPs, std::size(kMsaaLoadPs), &msaa_table);
    msaa_table.resources[0].sample_count = 2;
    const auto msaa_2x = recompile_graphics_shader_cached(
        ShaderProgramStage::Fragment, kMsaaLoadPs, std::size(kMsaaLoadPs), &msaa_table);
    stats = shader_recompile_cache_stats();
    CHECK(!msaa_4x.empty() && msaa_4x_again == msaa_4x && msaa_2x.empty() &&
              stats.misses == 2 && stats.hits == 1 && stats.entries == 2,
          "shader cache separates the supported 4x MSAA fetch from unsupported sample counts");

    // A FLAT-window's base push-constant index is embedded in the module. Keep it in the key even
    // though both variants use the same binding and exact fetch PC.
    clear_shader_recompile_cache();
    static const uint32_t kFlatCompute[] = {
        0xd70f6a01u, 0x00020c0cu, 0x50041af9u, 0x86860680u,
        0xdc200000u, 0x007d0001u, 0xbf810000u,
    };
    ComputeShaderConfig flat_config;
    flat_config.user_sgprs.resize(16);
    ShaderResource flat_window;
    flat_window.cls = ResourceClass::ConstantBuffer;
    flat_window.binding = 4;
    flat_window.fetch_pc = 4;
    flat_window.flat_base_sgpr = 12;
    ShaderResourceTable flat_table;
    flat_table.resources.push_back(flat_window);
    const auto flat_s12 = recompile_compute_shader_cached(
        kFlatCompute, std::size(kFlatCompute), &flat_table, flat_config);
    flat_table.resources[0].flat_base_sgpr = 10;
    const auto flat_s10 = recompile_compute_shader_cached(
        kFlatCompute, std::size(kFlatCompute), &flat_table, flat_config);
    stats = shader_recompile_cache_stats();
    CHECK(!flat_s12.empty() && !flat_s10.empty() && flat_s12 != flat_s10 &&
              stats.misses == 2 && stats.hits == 0,
          "compute cache separates the FLAT-window push-constant base SGPR");

    clear_shader_recompile_cache();
    // GFX9+ may install a vertex-fetch prolog separately from the main merged shader. Both immutable
    // allocations form one compiled program, so repeated pairs share an entry while changing only
    // the main allocation must miss even when the prolog and resource interface are unchanged.
    const uint32_t vertex_prolog[] = {
        0xBFA00003u, // s_setprio 3
        0xBE802006u, // s_setpc_b64 s[6:7]
    };
    std::vector<uint32_t> chain_main(kVs, kVs + std::size(kVs));
    uint64_t chain_first_identity = 0, chain_second_identity = 0;
    const SharedShaderWords chain_first = recompile_vertex_chain_cached_shared(
        vertex_prolog, std::size(vertex_prolog), chain_main.data(), chain_main.size(),
        &table, nullptr, &chain_first_identity);
    const SharedShaderWords chain_second = recompile_vertex_chain_cached_shared(
        vertex_prolog, std::size(vertex_prolog), chain_main.data(), chain_main.size(),
        &table, nullptr, &chain_second_identity);
    stats = shader_recompile_cache_stats();
    CHECK(chain_first && chain_first == chain_second && !chain_first->empty() &&
              chain_first_identity != 0 && chain_first_identity == chain_second_identity &&
              stats.misses == 1 && stats.hits == 1,
          "identical vertex prolog/main pairs share one compiled cache entry");

    // The factory returns an unfinished key: chained vertices, like ordinary graphics and compute,
    // attach the diagnostic address before hashing it. Relocated identical prologs must separate
    // while a selector is armed, and each finalized key must remain reusable in either call order.
    {
        const std::vector<uint32_t> relocated_prolog(
            vertex_prolog, vertex_prolog + std::size(vertex_prolog));
        char selector[32];
        std::snprintf(selector, sizeof selector, "0x%llx",
                      static_cast<unsigned long long>(
                          reinterpret_cast<uintptr_t>(vertex_prolog)));
        set_test_env("PROSPER_CFG_TRIP_BOUND", "4");
        set_test_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", selector);
        set_test_env("PROSPER_CFG_TRIP_BOUND_PHASE", "0");
        for (bool relocated_first : {false, true}) {
            clear_shader_recompile_cache();
            const uint32_t* first = relocated_first ? relocated_prolog.data() : vertex_prolog;
            const uint32_t* second = relocated_first ? vertex_prolog : relocated_prolog.data();
            uint64_t first_id = 0, second_id = 0, repeated_id = 0;
            const auto first_words = recompile_vertex_chain_cached_shared(
                first, std::size(vertex_prolog), chain_main.data(), chain_main.size(),
                &table, nullptr, &first_id);
            const auto second_words = recompile_vertex_chain_cached_shared(
                second, std::size(vertex_prolog), chain_main.data(), chain_main.size(),
                &table, nullptr, &second_id);
            const auto repeated_words = recompile_vertex_chain_cached_shared(
                first, std::size(vertex_prolog), chain_main.data(), chain_main.size(),
                &table, nullptr, &repeated_id);
            const auto selected_stats = shader_recompile_cache_stats();
            CHECK(first_words && second_words && !first_words->empty() && !second_words->empty() &&
                      first_id != 0 && second_id != 0 && first_id != second_id &&
                      repeated_words == first_words && repeated_id == first_id &&
                      selected_stats.misses == 2 && selected_stats.hits == 1,
                  "armed chained-vertex keys separate addresses and hit after finalization");
        }
        set_test_env("PROSPER_CFG_TRIP_BOUND", nullptr);
        set_test_env("PROSPER_CFG_TRIP_BOUND_PROGRAM", nullptr);
        set_test_env("PROSPER_CFG_TRIP_BOUND_PHASE", nullptr);
        // Restore the ordinary pair for the LDS and code-mutation arms below.
        clear_shader_recompile_cache();
        (void)recompile_vertex_chain_cached_shared(
            vertex_prolog, std::size(vertex_prolog), chain_main.data(), chain_main.size(),
            &table, nullptr, &chain_first_identity);
        stats = shader_recompile_cache_stats();
    }

    const uint64_t pre_lds_misses = stats.misses;
    uint64_t chain_lds_identity = 0;
    const SharedShaderWords chain_with_lds = recompile_vertex_chain_cached_shared(
        vertex_prolog, std::size(vertex_prolog), chain_main.data(), chain_main.size(),
        &table, nullptr, &chain_lds_identity, 4);
    stats = shader_recompile_cache_stats();
    CHECK(chain_with_lds && chain_lds_identity != chain_first_identity &&
              stats.misses == pre_lds_misses + 1,
          "graphics LDS allocation participates in chained shader cache identity");

    const uint64_t chain_misses = stats.misses;
    chain_main.insert(chain_main.begin(), 0xBFA00002u); // s_setprio 2: valid, distinct main bytes
    uint64_t changed_chain_identity = 0;
    const SharedShaderWords changed_chain = recompile_vertex_chain_cached_shared(
        vertex_prolog, std::size(vertex_prolog), chain_main.data(), chain_main.size(),
        &table, nullptr, &changed_chain_identity);
    stats = shader_recompile_cache_stats();
    CHECK(changed_chain && !changed_chain->empty() &&
              changed_chain_identity != chain_first_identity &&
              stats.misses == chain_misses + 1,
          "changing only the chained main shader invalidates compiled cache identity");

    clear_shader_recompile_cache();
    uint64_t normal_identity = 0, capture_identity = 0, repeated_capture_identity = 0;
    const auto normal_vertex = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &normal_identity);
    const auto capture_vertex = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &capture_identity, false, 0, true);
    const auto repeated_capture_vertex = recompile_graphics_shader_cached(
        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
        &repeated_capture_identity, false, 0, true);
    stats = shader_recompile_cache_stats();
    CHECK(!normal_vertex.empty() && !capture_vertex.empty() &&
              normal_vertex != capture_vertex && normal_identity != capture_identity &&
              capture_vertex == repeated_capture_vertex &&
              capture_identity == repeated_capture_identity &&
              stats.misses == 2 && stats.hits == 1 && stats.entries == 2,
          "geometry-probe vertex capture has a distinct, reusable cache identity");

    // A buffer-array selector is compiled into the access chain. Warm-cache hits must therefore
    // partition on its exact arity/source and on shared-contract admission; otherwise a mutated
    // selector or invalid entry can borrow the already-valid module without reaching the emitter.
    clear_shader_recompile_cache();
    static const uint32_t kArrayCompute[] = {
        0xe0300000u, 0x80000100u, // buffer_load_dword v1, off, s[0:3]
        0xbf810000u,
    };
    ShaderResource array_resource;
    array_resource.cls = ResourceClass::ConstantBuffer;
    array_resource.format = DataFormat::Uint32;
    array_resource.num_components = 1;
    array_resource.binding = 2;
    array_resource.stride = 4;
    array_resource.sgpr_base = 0;
    array_resource.table_index_count = 2;
    array_resource.table_entry_stride = 16;
    array_resource.table_index_sgpr = 8;
    array_resource.table_selector_mode = BufferTableSelectorMode::UserSgprIndex;
    auto array_entry = [](uint64_t address) {
        ShaderBufferTableEntry entry;
        entry.gpu_addr = address;
        entry.size = 16;
        entry.stride = 4;
        entry.vsharp = {
            static_cast<uint32_t>(address),
            static_cast<uint32_t>(address >> 32u) | (4u << 16u),
            4u,
            (20u << 12u) | 0xfacu,
        };
        return entry;
    };
    array_resource.table_entries = {array_entry(0x200000u), array_entry(0x201000u)};
    ShaderResourceTable array_table;
    array_table.resources.push_back(array_resource);
    ComputeShaderConfig array_config;
    array_config.user_sgprs.resize(9);
    array_config.local_x = array_config.local_y = array_config.local_z = 1;
    const auto array_s8 = recompile_compute_shader_cached(
        kArrayCompute, std::size(kArrayCompute), &array_table, array_config);
    ShaderResourceTable array_s7_table = array_table;
    array_s7_table.resources[0].table_index_sgpr = 7;
    const auto array_s7 = recompile_compute_shader_cached(
        kArrayCompute, std::size(kArrayCompute), &array_s7_table, array_config);
    ShaderResourceTable array_three_table = array_table;
    array_three_table.resources[0].table_index_count = 3;
    array_three_table.resources[0].table_entries.push_back(array_entry(0x202000u));
    const auto array_three = recompile_compute_shader_cached(
        kArrayCompute, std::size(kArrayCompute), &array_three_table, array_config);
    ShaderResourceTable invalid_array_table = array_table;
    invalid_array_table.resources[0].table_entries[1].vsharp[3] |= 1u << 19u;
    const auto invalid_array = recompile_compute_shader_cached(
        kArrayCompute, std::size(kArrayCompute), &invalid_array_table, array_config);
    const auto array_s8_again = recompile_compute_shader_cached(
        kArrayCompute, std::size(kArrayCompute), &array_table, array_config);
    stats = shader_recompile_cache_stats();
    CHECK(!array_s8.empty() && !array_s7.empty() && !array_three.empty() &&
              array_s8 != array_s7 && array_s8 != array_three && invalid_array.empty() &&
              array_s8_again == array_s8 && stats.misses == 4 && stats.hits == 1 &&
              stats.entries == 4,
          "buffer-array cache separates arity, selector source, and same-entry admission mutations");

    // ---- #3196: the dump names its program by guest ADDRESS, and can be narrowed to one ----
    //
    // Four arms, and each one fails against a different half of the change. The copies are heap
    // vectors rather than second static arrays on purpose: identical `.rodata` objects can be
    // folded to one symbol by the linker, and that would silently give two "different" programs the
    // same address -- an instrument that cannot fail for the reason it is testing.
    {
        const std::filesystem::path address_dir =
            prosper_test::test_scratch_dir() / "prosper-shader-dump-address";
        std::error_code address_ec;
        std::filesystem::remove_all(address_dir, address_ec);
        set_test_env("PROSPER_SHADER_DUMP_SUCCESS", address_dir.string().c_str());
        set_test_env("PROSPER_SHADER_DUMP_PROGRAM", nullptr);

        ShaderResource address_resource;
        address_resource.cls = ResourceClass::VertexBuffer;
        address_resource.format = DataFormat::Float32;
        address_resource.num_components = 3;
        address_resource.binding = 7;
        address_resource.stride = 12;
        address_resource.srt_offset = 0x20;
        address_resource.sgpr_base = 8;
        address_resource.fetch_pc = 4;
        address_resource.gpu_addr = 0x100000;
        address_resource.size = 4096;
        ShaderResourceTable address_table;
        address_table.resources.push_back(address_resource);

        const std::vector<uint32_t> vs_copy(std::begin(kVs), std::end(kVs));
        const std::vector<uint32_t> vs_other(std::begin(kVs), std::end(kVs));
        const std::vector<uint32_t> vs_malformed_arm(std::begin(kVs), std::end(kVs));
        auto address_needle = [](const std::vector<uint32_t>& words) {
            char text[64];
            std::snprintf(text, sizeof text, "_at_%016llx_",
                          static_cast<unsigned long long>(
                              reinterpret_cast<uintptr_t>(words.data())));
            return std::string(text);
        };
        auto address_spec = [](const std::vector<uint32_t>& words) {
            char text[64];
            std::snprintf(text, sizeof text, "0x%llx",
                          static_cast<unsigned long long>(
                              reinterpret_cast<uintptr_t>(words.data())));
            return std::string(text);
        };

        // 1. Unfiltered. These are byte-identical to kVs, which an earlier arm already dumped, so
        //    with the pre-#3196 hash-only dedup key this writes NOTHING; and with the pre-#3196
        //    filename it could not be told apart from kVs's own pair.
        recompile_graphics_shader_cached(ShaderProgramStage::Vertex, vs_copy.data(),
                                         vs_copy.size(), &address_table);
        CHECK(count_names_containing(address_dir, address_needle(vs_copy).c_str()) == 2,
              "a dumped shader is named by its guest code address (#3196)");

        // 2. Armed on an address this program does not have: withheld, and the directory does not
        //    grow. A selector that cannot withhold is not a selector.
        set_test_env("PROSPER_SHADER_DUMP_PROGRAM", "0xdeadbeef");
        const size_t before_withheld = count_extension(address_dir, ".bin") +
                                       count_extension(address_dir, ".spv");
        recompile_graphics_shader_cached(ShaderProgramStage::Vertex, vs_other.data(),
                                         vs_other.size(), &address_table);
        CHECK(prosper::gpu::shader_dump_program_filter().armed() &&
                  count_names_containing(address_dir, address_needle(vs_other).c_str()) == 0 &&
                  count_extension(address_dir, ".bin") + count_extension(address_dir, ".spv") ==
                      before_withheld,
              "PROSPER_SHADER_DUMP_PROGRAM withholds a program it does not name (#3196)");

        // 3. Armed on this program's own address: written. Note it is the SAME program the previous
        //    arm withheld -- which only works because the filter is consulted BEFORE the dedup set,
        //    so a withheld program is never recorded as already dumped.
        set_test_env("PROSPER_SHADER_DUMP_PROGRAM", address_spec(vs_other).c_str());
        recompile_graphics_shader_cached(ShaderProgramStage::Vertex, vs_other.data(),
                                         vs_other.size(), &address_table);
        CHECK(count_names_containing(address_dir, address_needle(vs_other).c_str()) == 2,
              "PROSPER_SHADER_DUMP_PROGRAM dumps exactly the program it names (#3196)");

        // 4. Malformed. It fails OPEN and says so: an empty dump directory would read as "that
        //    program never compiled", which is the false negative this change exists to remove.
        set_test_env("PROSPER_SHADER_DUMP_PROGRAM", "deadbeef");
        recompile_graphics_shader_cached(ShaderProgramStage::Vertex, vs_malformed_arm.data(),
                                         vs_malformed_arm.size(), &address_table);
        CHECK(!prosper::gpu::shader_dump_program_filter().armed() &&
                  count_names_containing(
                      address_dir, address_needle(vs_malformed_arm).c_str()) == 2,
              "a malformed PROSPER_SHADER_DUMP_PROGRAM arms nothing and dumps everything (#3196)");

        set_test_env("PROSPER_SHADER_DUMP_PROGRAM", nullptr);
        set_test_env("PROSPER_SHADER_DUMP_SUCCESS", nullptr);
        std::filesystem::remove_all(address_dir, address_ec);
    }

    // Concurrent cache scaling: multiple threads querying the cache concurrently under shared_lock
    {
        // Prime the cache with one entry so all worker lookups are warm hits under shared_lock
        uint64_t prime_id = 0;
        recompile_graphics_shader_cached(
            ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
            &prime_id);
        const auto before_concurrent = shader_recompile_cache_stats();
        constexpr int kWorkerThreads = 8;
        constexpr int kLookupsPerWorker = 50;
        std::vector<std::thread> workers;
        workers.reserve(kWorkerThreads);
        std::atomic<int> success_count{0};
        for (int i = 0; i < kWorkerThreads; ++i) {
            workers.emplace_back([&] {
                for (int j = 0; j < kLookupsPerWorker; ++j) {
                    uint64_t worker_id = 0;
                    const auto hit = recompile_graphics_shader_cached(
                        ShaderProgramStage::Vertex, kVs, std::size(kVs), &table, nullptr, nullptr,
                        &worker_id);
                    if (!hit.empty() && worker_id != 0) {
                        success_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
        }
        for (auto& w : workers) w.join();
        const auto after_concurrent = shader_recompile_cache_stats();
        CHECK(success_count.load() == kWorkerThreads * kLookupsPerWorker &&
                  after_concurrent.hits >= before_concurrent.hits + kWorkerThreads * kLookupsPerWorker,
              "concurrent reader lookups are race-free and hit the shader cache");
    }

    // #3130, and deliberately LAST. The arm above cannot show that the program address actually
    // ARRIVES -- a signature that accepts the diagnostic context and drops it on the floor passes it.
    // record_terminal_reject_reason() early-returns on a zero address, so while
    // recompile_fragment_impl hardcoded 0 no fragment rejection was recorded for ANY program and
    // last_terminal_reject_reason() came back empty for all of them. A terminal reject at a known
    // address is the observable that separates the two.
    //
    // It runs last because it is not side-effect free: recording a reject reason, and the
    // once-per-program-address consequence sets behind it, are process-global. Placed mid-suite it
    // reddened six unrelated compute-identity cases -- so this position is load-bearing, not
    // stylistic. Add new cases ABOVE it.
    constexpr uint64_t kFragmentDiagnosticAddress = 0x0000004321000000ull;
    const auto rejected_fragment = recompile_fragment(
        kPs, std::size(kPs), nullptr, nullptr, /*pcrel_dispatch_target=*/7u, nullptr, false,
        {RecompileDiagnosticStage::Fragment, kFragmentDiagnosticAddress});
    CHECK(rejected_fragment.empty() &&
              !last_terminal_reject_reason(kFragmentDiagnosticAddress).empty(),
          "a fragment terminal reject is recorded against its own program address (#3130)");

    if (failures) {
        std::printf("== FAIL: %d ==\n", failures);
        return 1;
    }
    std::printf("== PASS ==\n");
    return 0;
}
