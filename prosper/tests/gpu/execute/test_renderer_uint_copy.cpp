// #3407: a renderer-owned RGBA8 UNORM target sampled as UINT must preserve all byte values.
// Exercise the live compute backend, including pin release, typed sampling and guest writeback.
#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"
#include "shared/live/live_compute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <memory>

using namespace prosper::gpu;
static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s\n", m); ++failures; } } while (0)

int main() {
    constexpr uint32_t width = 256;
    std::vector<uint8_t> stale(width * 4, 0), output(width * 4, 0xee);
    auto expected = std::make_shared<std::vector<uint8_t>>(width * 4);
    const uint64_t address = reinterpret_cast<uint64_t>(stale.data());
    const auto& ctx = prosper::test::render_vk_ctx(); // publish before compute adopts a device
    if (!ctx.ok) return 1;

    ResolvedPipelineState state{};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.color_write_mask = 0; // retain the seeded pattern across the harmless draw
    prosper::test::BackendDraw draw;
    draw.vs.assign(std::begin(kTriVertSpv), std::end(kTriVertSpv));
    draw.fs.assign(std::begin(kTriFragSpv), std::end(kTriFragSpv));
    draw.ps = &state;
    draw.vcount = 3;
    prosper::test::BackendColorTarget target{address, false, true};

    unsigned reads = 0, imports = 0, releases = 0;
    bool transfer_source = true, matching_extent = true;
    set_live_target_query([&](uint64_t addr) { return addr == address; });
    set_live_target_reader([&](uint64_t addr, LiveTargetSnapshot& snapshot) {
        if (addr != address) return false;
        ++reads;
        snapshot.width = width; snapshot.height = 1;
        snapshot.format = LiveTargetPixelFormat::Rgba8Unorm;
        snapshot.pixels = expected;
        return true;
    });
    set_live_target_image_importer(
        [&](uint64_t addr, const LiveTargetImageRequest&, LiveTargetImageImport& result) {
            if (addr != address) return false;
            prosper::test::BackendPersistentResourceGuard guard;
            auto* image = prosper::test::find_persistent_color_target(
                address, width, 1, VK_FORMAT_R8G8B8A8_UNORM);
            if (!image || !prosper::test::pin_persistent_color_target(
                    address, width, 1, VK_FORMAT_R8G8B8A8_UNORM)) return false;
            ++imports;
            result.width = matching_extent ? width : width / 2;
            result.height = 1;
            result.format = LiveTargetPixelFormat::Rgba8Unorm;
            result.image = image->image; result.device = ctx.dev;
            result.layout = image->layout;
            result.transfer_src = transfer_source;
            return true;
        },
        [&](uint64_t addr) {
            CHECK(addr == address, "release names the pinned renderer target");
            ++releases;
            prosper::test::BackendPersistentResourceGuard guard;
            prosper::test::unpin_persistent_color_target(
                address, width, 1, VK_FORMAT_R8G8B8A8_UNORM);
        });

    std::vector<uint32_t> indices(width), dummy(4, 0);
    for (uint32_t i = 0; i < width; ++i) indices[i] = i;
    ShaderResourceTable resources;
    for (uint32_t binding = 0; binding < 4; ++binding) {
        ShaderResource buffer{};
        buffer.cls = ResourceClass::ConstantBuffer;
        buffer.binding = binding;
        buffer.gpu_addr = reinterpret_cast<uint64_t>(binding ? dummy.data() : indices.data());
        buffer.size = binding ? 16 : width * sizeof(uint32_t);
        resources.resources.push_back(buffer);
    }
    for (uint32_t binding : {4u, 5u}) {
        ShaderResource image{};
        image.cls = binding == 4 ? ResourceClass::Texture : ResourceClass::StorageImage;
        image.binding = binding; image.sgpr_base = binding == 4 ? 0 : 8;
        image.img_dim = 1; image.width = width; image.height = image.depth = 1;
        image.format = DataFormat::Uint8; image.num_components = 4;
        image.gpu_addr = binding == 4 ? address : reinterpret_cast<uint64_t>(output.data());
        image.size = width * 4;
        for (uint32_t c = 0; c < 4; ++c) image.swizzle[c] = 4 + c;
        resources.resources.push_back(image);
    }
    // v4 = input[gid] (x), v5 = 0; image_load RGBA from s[0:7]; image_store to s[8:15].
    const uint32_t code[] = {
        0x7e080300u, 0x7e0a0280u, 0xf0000f08u, 0x00000004u, 0xbf8c3f70u,
        0xf0200f08u, 0x00020004u, 0xbf810000u,
    };
    ComputeItem item;
    item.spirv = recompile_valu(code, std::size(code), 1, 0, &resources);
    CHECK(!item.spirv.empty(), "UINT image copy kernel recompiles");
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.launch.threads_x = width;
    item.launch.local_x = 64; item.launch.groups_x = width / 64;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_y = item.launch.groups_z = 1;
    item.code_addr = 0x3407;

    for (unsigned round = 0; round < 4; ++round) {
        for (uint32_t i = 0; i < width * 4; ++i)
            (*expected)[i] = static_cast<uint8_t>((i / 4) * 37 + (i % 4) * 23 + round * 113 + 11);
        const auto rendered = prosper::test::render_draws_rgba(
            {draw}, width, 1, expected->data(), nullptr, false, &target);
        CHECK(rendered == *expected, "renderer holds the exact nontrivial RGBA8 pattern");
        // First two rounds replace content at the same identity. The remaining arms require the
        // old CPU route: neither a missing transfer usage nor a scaled image is an exact copy.
        transfer_source = round != 2;
        matching_extent = round != 3;
        std::fill(output.begin(), output.end(), 0xee);
        const unsigned reads_before = reads;
        const unsigned imports_before = imports;
        CHECK(prosper::frontend::execute_live_compute_items({item}), "live UINT copy executes");
        CHECK(output == *expected, "all 256 byte values survive UINT sampling and guest writeback");
        CHECK(imports > imports_before && imports == releases, "every acquired image pin is released");
        CHECK(round < 2 ? reads == reads_before : reads == reads_before + 1,
              "eligible copies avoid CPU readback; incompatible imports use the snapshot fallback");
    }
    // Two descriptors can pin the same renderer image. Check folded UINT aliases, then each
    // ordering of a direct UNORM borrow beside the UINT copy. The latter must preserve GENERAL
    // while the direct borrower owns it, then restore the renderer's original layout on release.
    transfer_source = matching_extent = true;
    for (unsigned order = 0; order < 3; ++order) {
        auto paired = resources;
        auto second = paired.resources[4];
        second.binding = 6; second.sgpr_base = 16;
        if (order == 1) paired.resources[4].format = DataFormat::Unorm8;
        if (order == 2) second.format = DataFormat::Unorm8;
        paired.resources.push_back(second);
        const uint32_t paired_code[] = {
            0x7e080300u, 0x7e0a0280u,
            0xf0000f08u, order == 2 ? 0x00040004u : 0x00000004u, 0xbf8c3f70u,
            0xf0000f08u, order == 2 ? 0x00000004u : 0x00040004u, 0xbf8c3f70u,
            0xf0200f08u, 0x00020004u, 0xbf810000u,
        };
        item.spirv = recompile_valu(paired_code, std::size(paired_code), 1, 0, &paired);
        CHECK(!item.spirv.empty(), "paired UNORM/UINT copy kernel recompiles");
        item.resources = std::make_shared<ShaderResourceTable>(paired);
        const unsigned reads_before = reads, imports_before = imports;
        std::fill(output.begin(), output.end(), 0xee);
        CHECK(prosper::frontend::execute_live_compute_items({item}), "paired image imports execute");
        CHECK(output == *expected, "alias folding and mixed view order preserve UINT values");
        CHECK(reads == reads_before && imports == imports_before + 2 && imports == releases,
              "both borrowed descriptors release their pins without a CPU snapshot");
        std::vector<uint8_t> retained;
        std::string error;
        prosper::test::BackendPersistentResourceGuard guard;
        CHECK(prosper::test::readback_persistent_color_target(
                  address, width, 1, VK_FORMAT_R8G8B8A8_UNORM, retained, error) && retained == *expected,
              "the renderer source retains its pixels and usable layout after both imports");
    }
    set_live_target_image_importer({}, {});
    set_live_target_reader({});
    set_live_target_query({});
    std::printf("renderer UINT copy: %d failures\n", failures);
    return failures ? 1 : 0;
}
