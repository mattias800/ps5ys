// Metadata-only storage alias access union. Expectations are hand-built member sets and access
// verdicts, never calls to the same identity helper used by the planner as a second oracle.
#include "shared/compute/storage_image_alias_plan.hpp"

#include <array>
#include <cstdio>
#include <initializer_list>
#include <vector>

using namespace prosper::gpu;
using prosper::frontend::StorageImageAliasPlan;
using prosper::frontend::plan_storage_image_aliases;

static int failures = 0;
static const char* context = "initial";
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %s:%d [%s]: %s\n", __FILE__, __LINE__, context, #condition); \
    ++failures; } } while (0)

static ShaderResource resource(uint32_t binding, uint64_t address = 0x10000) {
    ShaderResource r;
    r.cls = ResourceClass::StorageImage;
    r.binding = binding;
    r.gpu_addr = address;
    r.size = 256;
    r.width = r.height = 8;
    r.depth = 1;
    r.img_dim = 1;
    r.format = DataFormat::Unorm8;
    r.num_components = 4;
    r.tile_mode = 27;
    return r;
}

static SpirvDescriptorBinding descriptor(uint32_t binding, bool read, bool write) {
    SpirvDescriptorBinding d;
    d.binding = binding;
    d.kind = SpirvDescriptorKind::StorageImage;
    d.stage = SpirvShaderStage::Compute;
    d.image_dim = 1;
    d.storage_float = true;
    d.image_numeric_class = SpirvImageNumericClass::Float;
    d.readable = read;
    d.writable = write;
    return d;
}

static void group(const StorageImageAliasPlan& plan, size_t index, size_t owner,
                  std::initializer_list<uint32_t> members,
                  bool read, bool write, bool atomic, bool discard) {
    CHECK(index < plan.groups.size());
    if (index >= plan.groups.size()) return;
    const auto& g = plan.groups[index];
    CHECK(g.owner_index == owner);
    CHECK(g.bindings == std::vector<uint32_t>(members));
    CHECK(g.readable == read);
    CHECK(g.writable == write);
    CHECK(g.atomic == atomic);
    CHECK(g.can_discard_seed() == discard);
}

static ShaderResourceTable pair_resources() {
    ShaderResourceTable table;
    table.resources = {resource(5), resource(26)};
    return table;
}

template<class Mutate>
static void resource_split(const char* label, Mutate mutate) {
    context = label;
    auto table = pair_resources();
    mutate(table.resources[1]);
    const std::array ds{descriptor(5, false, true), descriptor(26, true, false)};
    const auto plan = plan_storage_image_aliases(ds, table);
    CHECK(plan.valid);
    CHECK(plan.groups.size() == 2);
    CHECK((plan.group_for_image == std::vector<size_t>{0, 1}));
    group(plan, 0, 0, {5}, false, true, false, true);
    group(plan, 1, 1, {26}, true, false, false, false);
}

int main() {
    auto table = pair_resources();
    // Reverse descriptor visitation, but not the resource-table order: lookup is by binding.
    for (bool reverse : {false, true}) {
        context = reverse ? "reader then writer" : "writer then reader";
        std::array ds{descriptor(5, false, true), descriptor(26, true, false)};
        if (reverse) std::swap(ds[0], ds[1]);
        auto plan = plan_storage_image_aliases(ds, table);
        CHECK(plan.valid);
        CHECK(plan.groups.size() == 1);
        CHECK((plan.group_for_image == std::vector<size_t>{0, 0}));
        group(plan, 0, 0, {5, 26}, true, true, false, false);

        context = reverse ? "atomic then writer" : "writer then atomic";
        ds = {descriptor(5, false, true), descriptor(26, false, true)};
        ds[1].atomic_access = true;
        if (reverse) std::swap(ds[0], ds[1]);
        plan = plan_storage_image_aliases(ds, table);
        CHECK(plan.groups.size() == 1);
        group(plan, 0, 0, {5, 26}, false, true, true, false);
    }

    context = "write-only union and distinct reader";
    table.resources.push_back(resource(42, 0x20000));
    std::array ds{descriptor(26, false, true), descriptor(5, false, true),
                  descriptor(42, true, false)};
    auto plan = plan_storage_image_aliases(ds, table);
    CHECK(plan.valid);
    CHECK((plan.group_for_image == std::vector<size_t>{0, 0, 1}));
    group(plan, 0, 0, {5, 26}, false, true, false, true);
    group(plan, 1, 2, {42}, true, false, false, false);

    context = "sampled image is not a storage-group reader";
    ds[2].kind = SpirvDescriptorKind::CombinedImageSampler;
    table.resources[2].gpu_addr = 0x10000;
    plan = plan_storage_image_aliases(ds, table);
    CHECK(plan.valid);
    CHECK(plan.groups.size() == 1);
    CHECK((plan.group_for_image == std::vector<size_t>{0, 0, SIZE_MAX}));
    group(plan, 0, 0, {5, 26}, false, true, false, true);

    context = "read-only and inactive groups cannot discard";
    std::array only{descriptor(5, true, false)};
    group(plan_storage_image_aliases(only, table), 0, 0, {5}, true, false, false, false);
    only[0].readable = false;
    group(plan_storage_image_aliases(only, table), 0, 0, {5}, false, false, false, false);

    // Distinct full-view fields must not turn a separate reader into this writer's alias.
#define SPLIT(field, value) resource_split(#field, [](ShaderResource& r) { r.field = value; })
    SPLIT(gpu_addr, 0x20000);
    SPLIT(size, 512);
    SPLIT(width, 16);
    SPLIT(height, 16);
    SPLIT(depth, 2);
    SPLIT(format, DataFormat::Float32);
    SPLIT(num_components, 2);
    SPLIT(tile_mode, 0);
    SPLIT(img_dim, 2);
    SPLIT(layer_stride_bytes, 1024);
    SPLIT(layer_mip_offset_bytes, 128);
    SPLIT(in_mip_tail, true);
    SPLIT(mip_tail_x, 1);
    SPLIT(mip_tail_y, 1);
    SPLIT(declared_mip_levels, 2);
    SPLIT(srgb, true);
    SPLIT(host_data_size, 256);
    SPLIT(host_data_prefix_bytes, 64);
    SPLIT(max_uncompressed_block_size, 1);
    SPLIT(max_compressed_block_size, 1);
    SPLIT(meta_pipe_aligned, true);
    SPLIT(write_compress_enabled, true);
    SPLIT(compression_enabled, true);
    SPLIT(alpha_is_on_msb, true);
    SPLIT(color_transform, true);
    SPLIT(metadata_addr, 0x40000);
    SPLIT(dcc_metadata_size, 64);
    SPLIT(dcc_metadata_host_data_size, 64);
#undef SPLIT
    std::array<uint8_t, 512> blob_a{}, blob_b{};
    resource_split("DCC host backing identity", [&](ShaderResource& r) {
        r.dcc_metadata_host_data = blob_a.data();
    });

    context = "same guest range in independently owned capture blobs";
    table = pair_resources();
    table.resources[0].host_data = blob_a.data() + 64;
    table.resources[1].host_data = blob_b.data() + 64;
    for (auto& r : table.resources) { r.host_data_size = 256; r.host_data_prefix_bytes = 64; }
    const std::array pair{descriptor(5, false, true), descriptor(26, true, false)};
    plan = plan_storage_image_aliases(pair, table);
    CHECK(plan.groups.size() == 1);
    group(plan, 0, 0, {5, 26}, true, true, false, false);
    context = "capture replacement owns a different prefix";
    table.resources[1].host_data_prefix_bytes = 0;
    plan = plan_storage_image_aliases(pair, table);
    CHECK(plan.groups.size() == 2);
    CHECK((plan.group_for_image == std::vector<size_t>{0, 1}));

    context = "arrayed reflection changes realized layers";
    table = pair_resources();
    for (auto& r : table.resources) { r.img_dim = 5; r.depth = 3; }
    auto shape_pair = pair;
    shape_pair[1].image_arrayed = true;
    plan = plan_storage_image_aliases(shape_pair, table);
    CHECK(plan.groups.size() == 2);
    CHECK((plan.group_for_image == std::vector<size_t>{0, 1}));
    shape_pair[0].image_arrayed = true;
    CHECK(plan_storage_image_aliases(shape_pair, table).groups.size() == 1);

    context = "3D shape is metadata, not allocation validation";
    auto volume = resource(5);
    volume.img_dim = 2; volume.depth = 3;
    const auto shape = prosper::frontend::compute_image_alias_shape(volume, pair[0]);
    CHECK(shape.storage && shape.texel_depth == 3 && shape.array_layers == 1);

    // Single-level provenance is deliberately irrelevant; a real mip chain makes it identity.
    for (bool multilevel : {false, true}) {
        context = multilevel ? "mip-chain provenance splits" : "single-level provenance ignored";
        table = pair_resources();
        for (auto& r : table.resources) r.declared_mip_levels = multilevel ? 2 : 1;
        table.resources[1].mip_chain_element_width = 32;
        plan = plan_storage_image_aliases(pair, table);
        CHECK(plan.groups.size() == (multilevel ? 2u : 1u));
    }

    context = "reflected format must not hide a late-folded reader";
    table = pair_resources();
    auto reflected_pair = pair;
    reflected_pair[1].storage_image_format = 33; // R32ui versus the first binding's Unknown.
    reflected_pair[1].storage_float = false;
    reflected_pair[1].image_numeric_class = SpirvImageNumericClass::Uint;
    plan = plan_storage_image_aliases(reflected_pair, table);
    CHECK(plan.groups.size() == 1);
    group(plan, 0, 0, {5, 26}, true, true, false, false);

    context = "member set survives address movement, changes on join and split";
    table.resources.push_back(resource(42, 0x20000));
    ds = {descriptor(26, false, true), descriptor(5, false, true), descriptor(42, false, true)};
    plan = plan_storage_image_aliases(ds, table);
    group(plan, 0, 0, {5, 26}, false, true, false, true);
    group(plan, 1, 2, {42}, false, true, false, true);
    for (auto& r : table.resources) r.gpu_addr += 0x80000;
    plan = plan_storage_image_aliases(ds, table);
    group(plan, 0, 0, {5, 26}, false, true, false, true);
    group(plan, 1, 2, {42}, false, true, false, true);
    table.resources[2].gpu_addr = table.resources[0].gpu_addr;
    plan = plan_storage_image_aliases(ds, table);
    CHECK(plan.groups.size() == 1);
    group(plan, 0, 0, {5, 26, 42}, false, true, false, true);
    table.resources[1].gpu_addr += 0x10000;
    plan = plan_storage_image_aliases(ds, table);
    CHECK((plan.group_for_image == std::vector<size_t>{0, 1, 1}));
    group(plan, 0, 0, {26}, false, true, false, true);
    group(plan, 1, 1, {5, 42}, false, true, false, true);

    context = "missing storage resource fails closed without hiding valid groups";
    table = pair_resources();
    table.resources.pop_back();
    plan = plan_storage_image_aliases(pair, table);
    CHECK(!plan.valid);
    CHECK((plan.group_for_image == std::vector<size_t>{0, SIZE_MAX}));
    group(plan, 0, 0, {5}, false, true, false, true);
    auto missing_first = pair;
    std::swap(missing_first[0], missing_first[1]);
    plan = plan_storage_image_aliases(missing_first, table);
    CHECK(!plan.valid);
    CHECK((plan.group_for_image == std::vector<size_t>{SIZE_MAX, 0}));
    group(plan, 0, 1, {5}, false, true, false, true);
    missing_first[0].kind = SpirvDescriptorKind::CombinedImageSampler;
    CHECK(plan_storage_image_aliases(missing_first, table).valid);
    const std::array<SpirvDescriptorBinding, 0> empty{};
    plan = plan_storage_image_aliases(empty, table);
    CHECK(plan.valid && plan.groups.empty() && plan.group_for_image.empty());

    std::printf("storage_image_alias_plan: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
