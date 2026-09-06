#pragma once

#include <vulkan/vulkan.h>

#include <cstdio>

namespace prosper::frontend {

// Runtime backends share one floor. Standalone Vulkan compatibility probes may deliberately
// request older versions; they do not create devices that the runtime adopts.
inline constexpr uint32_t kVulkanRuntimeVersion = VK_API_VERSION_1_4;

constexpr bool vulkan_runtime_version_supported(uint32_t version) {
    return VK_API_VERSION_VARIANT(version) == 0 && version >= kVulkanRuntimeVersion;
}

inline bool require_vulkan_runtime_loader(
    const char* owner, PFN_vkGetInstanceProcAddr get_proc = vkGetInstanceProcAddr) {
    const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        get_proc(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    // The function is absent on a Vulkan 1.0 loader. Do not call a newer entry point directly
    // before checking it, or the diagnostic itself depends on the version it needs to reject.
    uint32_t version = VK_API_VERSION_1_0;
    const VkResult result = enumerate ? enumerate(&version) : VK_SUCCESS;
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "[%s] Vulkan loader version query failed (%d)\n", owner, result);
        return false;
    }
    if (!vulkan_runtime_version_supported(version)) {
        std::fprintf(stderr, "[%s] Vulkan 1.4 required; loader reports %u.%u.%u\n", owner,
                     VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version),
                     VK_API_VERSION_PATCH(version));
        return false;
    }
    return true;
}

inline void log_vulkan_runtime_device(const char* owner, VkPhysicalDevice device,
                                      const VkPhysicalDeviceProperties& properties) {
    // Support is queried on the selected device, not inferred from the loader version or
    // another ICD (software devices can advertise features a hardware GPU lacks).
    VkPhysicalDeviceHostImageCopyFeatures host_copy{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &host_copy;
    vkGetPhysicalDeviceFeatures2(device, &features);
    std::fprintf(stderr, "[%s] Vulkan runtime 1.4, device API %u.%u.%u; "
                         "hostImageCopy supported=%u (not enabled)\n", owner,
                 VK_API_VERSION_MAJOR(properties.apiVersion),
                 VK_API_VERSION_MINOR(properties.apiVersion),
                 VK_API_VERSION_PATCH(properties.apiVersion), host_copy.hostImageCopy);
}

} // namespace prosper::frontend
