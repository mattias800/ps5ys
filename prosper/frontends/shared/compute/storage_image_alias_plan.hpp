#pragma once

#include "gpu/resources/image_identity.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace prosper::frontend {

// Metadata only: this is the shape used by live compute before allocating an image. Validation
// still belongs to the materializer; an invalid descriptor must not become valid by joining a group.
inline prosper::gpu::ComputeImageViewShape compute_image_alias_shape(
    const prosper::gpu::ShaderResource& resource,
    const prosper::gpu::SpirvDescriptorBinding& descriptor) {
    return {
        descriptor.kind == prosper::gpu::SpirvDescriptorKind::StorageImage,
        resource.img_dim == 2 ? resource.depth : 1u,
        resource.img_dim == 5 && descriptor.image_dim == 1u && descriptor.image_arrayed
            ? resource.depth : 1u,
    };
}

struct StorageImageAliasGroup {
    size_t owner_index = SIZE_MAX;
    std::vector<uint32_t> bindings;
    bool readable = false;
    bool writable = false;
    bool atomic = false;

    bool can_discard_seed() const { return writable && !readable && !atomic; }
};

struct StorageImageAliasPlan {
    bool valid = true;
    // Non-storage descriptors have no group. Indices refer to the supplied descriptor order.
    std::vector<size_t> group_for_image;
    std::vector<StorageImageAliasGroup> groups;
};

// Seeding must account for EVERY descriptor that will share the canonical image, before that
// image is uploaded or poisoned. This follows the existing late storage-alias identity, not the
// narrower early-fold optimization: storage never imports renderer/depth/value-reuse images, so
// its late same_backing_representation is true. Reflection-format differences must not hide a
// reader here when the late path can still fold it. This plan does not change actual folding.
inline StorageImageAliasPlan plan_storage_image_aliases(
    std::span<const prosper::gpu::SpirvDescriptorBinding> descriptors,
    const prosper::gpu::ShaderResourceTable& resources) {
    StorageImageAliasPlan plan;
    plan.group_for_image.assign(descriptors.size(), SIZE_MAX);
    std::vector<const prosper::gpu::ShaderResource*> resolved(descriptors.size(), nullptr);
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto& descriptor = descriptors[i];
        if (descriptor.kind != prosper::gpu::SpirvDescriptorKind::StorageImage) continue;
        const auto* resource = resources.by_binding(descriptor.binding);
        if (!resource) {
            plan.valid = false;
            continue;
        }
        resolved[i] = resource;
        const auto shape = compute_image_alias_shape(*resource, descriptor);
        size_t group_index = 0;
        for (; group_index < plan.groups.size(); ++group_index) {
            const size_t owner = plan.groups[group_index].owner_index;
            if (prosper::gpu::shader_resource_same_view(
                    *resolved[owner], *resource,
                    compute_image_alias_shape(*resolved[owner], descriptors[owner]), shape, true))
                break;
        }
        if (group_index == plan.groups.size()) {
            plan.groups.emplace_back();
            plan.groups.back().owner_index = i;
        }
        plan.group_for_image[i] = group_index;
        auto& group = plan.groups[group_index];
        group.bindings.push_back(descriptor.binding);
        group.readable |= descriptor.readable;
        group.writable |= descriptor.writable;
        group.atomic |= descriptor.atomic_access;
    }
    // Coverage is a property of this member set, not of transient guest addresses. Removing a
    // writer can turn Full into Partial/None, and adding a writer can invalidate a None verdict.
    for (auto& group : plan.groups) std::sort(group.bindings.begin(), group.bindings.end());
    return plan;
}

} // namespace prosper::frontend
