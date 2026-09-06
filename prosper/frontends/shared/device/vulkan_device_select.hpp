#pragma once

#include <vulkan/vulkan.h>
#include "vulkan_runtime.hpp"

#include <cstdint>
#include <vector>

namespace prosper::frontend {

struct VulkanDeviceSelection {
    VkPhysicalDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = UINT32_MAX;
    VkPhysicalDeviceProperties properties{};
};

constexpr int vulkan_device_type_rank(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 5;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 4;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_OTHER: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
    default: return 0;
    }
}

static_assert(vulkan_device_type_rank(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) >
              vulkan_device_type_rank(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU));
static_assert(vulkan_device_type_rank(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) >
              vulkan_device_type_rank(VK_PHYSICAL_DEVICE_TYPE_CPU));

// Vulkan does not promise that physical devices are enumerated in performance order. Prefer a
// hardware GPU when one supports the runtime version, queue and robustness contract, retaining
// llvmpipe and other CPU implementations as a headless fallback.
inline VulkanDeviceSelection select_vulkan_device(
    const std::vector<VkPhysicalDevice>& devices, VkQueueFlags required_queue_flags) {
    VulkanDeviceSelection best;
    int best_rank = -1;
    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        if (!vulkan_runtime_version_supported(properties.apiVersion)) continue;
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceFeatures(device, &features);
        if (!features.robustBufferAccess) continue;

        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());
        uint32_t queue_family = UINT32_MAX;
        for (uint32_t i = 0; i < family_count; ++i) {
            if ((families[i].queueFlags & required_queue_flags) == required_queue_flags) {
                queue_family = i;
                break;
            }
        }
        if (queue_family == UINT32_MAX) continue;

        const int rank = vulkan_device_type_rank(properties.deviceType);
        if (rank > best_rank) {
            best = {device, queue_family, properties};
            best_rank = rank;
        }
    }
    return best;
}

inline const char* vulkan_device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
    default: return "other";
    }
}

} // namespace prosper::frontend
