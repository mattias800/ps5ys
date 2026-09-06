// prosper-app (P0a) — the OS-integration frontend: an SDL3 window + Vulkan swapchain that presents
// the frame prosper's renderer hands to the present layer. See
// docs/FRONTEND_APP.md and issue #164.
//
// P0a scope (this file): the whole present half, decoupled from the guest boot. It pulls finished
// frames from prosper_core's present layer and blits them to a real swapchain,
// and handles SDL_QUIT/Esc via the shared stop request. run_entry does not consume that request yet,
// so a booted guest uses direct process exit after the present loop (issue #352). To verify the
// pipeline without a game dump the app can FEED the present layer a synthetic animated pattern
// (--test-pattern): frame -> present_write_frame -> shared frame lease -> swapchain, exactly the path
// a real guest frame takes. P0b wires the actual guest boot in front of this (the same present
// path, no changes here).
//
// Real game boots normally adopt the renderer's Vulkan device and pass its front image directly to
// the swapchain. Test-pattern boots, an explicit override, or failed adoption retain the original
// two-device path, where frames cross as shared immutable CPU pixels.
#include "gpu/present/videoout_present.hpp"   // present_acquire_rendered_frame / present_write_frame
#include "gpu/execute/gpu_execute.hpp"         // shared_vulkan_context / gpu-present activation (#1270)
#include "gpu/capture/gpu_capture.hpp"         // request_interactive_gpu_capture (F9 frame grab)
#include "gpu/timeline/gpu_timeline.hpp"        // request_interactive_capture_bundle (F9 whole-frame grab)
#include "capture_schedule.hpp"        // exact host-frame screenshot calibration trigger
#include "shared/present/present_blit.hpp"           // GPU scanout handoff: acquire/release the renderer's front image
#include "shared/present/present_blit_policy.hpp"    // reject stale CPU/GPU representations of guest flips
#include "shared/device/vulkan_runtime.hpp"
#include "hle/sync/sync_futex.hpp"         // dump_guest_sync_trace (PROSPER_SYNC_RING deadlock history)
#include "host/platform/lifecycle.hpp"          // frontend-owned stop/pause gates
#include "host/platform/gpu_submit_gate.hpp"     // #3225: drain guest GPU submits before _Exit
#include "host/image/boot_program.hpp"       // boot_program (shared guest-boot path, also used by boot_trace)
#include "host/image/exec_image.hpp"         // run_entry
#include "host/memory/guest_write_watch.hpp"  // flush dmem writer diagnostic before deliberate _Exit
#include "loader/linker.hpp"           // Program
#include "input/pad.hpp"               // keyboard -> libScePad (HostPadState / PadBackend)
#include "pad_overlay.hpp"              // keyboard pad 0 composed over the physical controller backend
#include "keyboard_pad_map.hpp"         // which key is which pad control (#2234)
#include "hle/input/ime_input.hpp"           // #1093: forward host keyboard keys to the guest IME path
#include "present_mode.hpp"             // explicit swapchain latency/vsync policy, pure regression seam
#include "hle/graphics/display_mode.hpp" // #3017: validate --display-mode against the one policy parser
#include "present_policy.hpp"           // bounded-acquire present classification (#1182), pure seam
#include "window_controls.hpp"           // debounced app-window shortcuts, pure regression seam
#include "game_path.hpp"                 // dropped/picked path -> app0 root + open action, pure seam
#include "game_library.hpp"              // scan a games dir -> titles + metadata, pure seam
#include "frame_grab_naming.hpp"         // one stamp + one exclusively claimed name pair per F9 grab
#include "shared/perf/performance_capture.hpp"        // bounded F8 pre/post performance artifact
#include "performance_capture_schedule.hpp" // unattended elapsed-time trigger for the same artifact
#include "shared/diagnostics/renderdoc_capture.hpp" // frame-aimed RenderDoc capture (#3321)
#include "app_config.hpp"                // persisted settings (games_dir), pure seam
// The --fps HUD is NOT part of the library view and is not guarded by its macro: `Vk::overlay` and
// every use site are unconditional, so the object and its header live outside PROSPER_HAVE_LIBRARY_UI
// too. They briefly did not, which compiled only because CMake defines that macro unconditionally
// for this target -- a latent break rather than a working arrangement.
#include "gpu/present/present_frame_rate.hpp"   // distinct-guest-frame rate (NOT a present rate)
#include "fps_overlay.hpp"               // --fps: the ImGui HUD drawn OVER a running title
#include "fps_hud.hpp"                   // ...and what it says, kept pure and unit-tested
#ifdef PROSPER_HAVE_LIBRARY_UI
#include "library_ui.hpp"                // the ImGui library grid drawn while no game is running
#endif
#ifdef PROSPER_HAVE_LIVE_RENDERER
#include "shared/live/live_renderer.hpp"           // shared DrawItem->Vulkan compositor (register_live_renderer)
#include "shared/live/live_compute.hpp"            // flush compute-timing selector summary before _Exit
#endif
#ifdef PROSPER_AUDIO_SDL3
#include "audio_sdl3.hpp"              // install_sdl3_audio_sink (route sceAudioOut to the host)
#endif

// Default playback volume for prosper-app, as a percent.
//
// UNITY, deliberately. prosper is a compatibility layer, so the title's own mix is the correct
// output and anything else is us editing it. This shipped as 25 in #2411 -- a bring-up default that
// made distorted audio bearable while iterating -- and that reasoning does not survive contact with
// a release: the AppImage and the tarball carry this value, and a user who finds prosper quieter
// than their console has no way to know we chose that for them. Attenuation during bring-up belongs
// on the command line (`--volume 25`), not in the default every user inherits.
static constexpr int kDefaultVolumePercent = 100;
static int g_volume_percent = kDefaultVolumePercent;   // set by --volume before backends install
#include "snap_author.hpp"            // human-authored render snapshots (F6/F7), always available
#ifdef PROSPER_AUDIO_FFMPEG
#include "ajm_ffmpeg.hpp"              // install AJM MP3 decoder before guest instance creation
#endif
#ifdef PROSPER_PAD_SDL3
#include "pad_sdl3.hpp"                // install_sdl3_pad_backend (route a host controller to libScePad)
#endif
#ifdef PROSPER_HAVE_DIALOG_SDL3
#include "dialog_sdl3.hpp"             // install_sdl3_platform_ui (real SDL message boxes for dialogs)
#endif
#ifdef PROSPER_VIDEO_MF
#include "media_foundation_backend.hpp" // native Windows AvPlayer demux + hardware decode
#endif
#ifdef PROSPER_VIDEO_VAAPI
#include "vaapi_backend.hpp"            // native Linux FFmpeg demux + VA-API hardware decode
#endif

#include <SDL3/SDL.h>
#include <SDL3/SDL_dialog.h>           // SDL_ShowOpenFolderDialog: the host's native folder picker
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>     // strict PROSPER_CAPTURE_BUNDLE_MAX_MB parse (#1587)
#include <climits>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <sys/stat.h>                  // the host filesystem probe behind resolve_app0_root()
#include <filesystem>                  // directory listing behind the library scan

// Starting the replacement process for a second title (#1469): CreateProcess on Windows,
// posix_spawn elsewhere. Guarded the same way as the rest of the tree so windows.h cannot
// redefine std::max.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <spawn.h>                     // posix_spawn: reports exec failure without forking the guest
extern char** environ;                 // the child inherits this process's environment
#ifdef __APPLE__
#include <mach-o/dyld.h>               // _NSGetExecutablePath (macOS has no /proc/self/exe)
#endif
#endif

using namespace prosper;

namespace {

bool set_environment(const char* name, const char* value) {
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}
bool clear_environment(const char* name) {
#ifdef _WIN32
    return _putenv_s(name, "") == 0;   // MSVC/MinGW: an empty value REMOVES the variable
#else
    return unsetenv(name) == 0;
#endif
}

// ---- tiny Vulkan error helper -----------------------------------------------------------------
#define VKCHECK(x, msg) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[app] Vulkan error %d at %s\n", (int)_r, msg); return false; } } while (0)

struct Vk {
    VkInstance       instance = VK_NULL_HANDLE;
    VkSurfaceKHR     surface  = VK_NULL_HANDLE;
    VkPhysicalDevice phys     = VK_NULL_HANDLE;
    VkDevice         device   = VK_NULL_HANDLE;
    uint32_t         qfamily  = 0;
    VkQueue          queue    = VK_NULL_HANDLE;

    VkSwapchainKHR   swapchain = VK_NULL_HANDLE;
    VkFormat         scFormat  = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D       scExtent  {};
    std::vector<VkImage> scImages;

    // A host-visible staging buffer + a device-local staging image; each frame we copy the guest
    // RGBA into the buffer, copy buffer->image, then BLIT (scaling) image->swapchain image.
    VkBuffer        stageBuf   = VK_NULL_HANDLE;
    VkDeviceMemory  stageMem   = VK_NULL_HANDLE;
    void*           stageMapped = nullptr;
    VkDeviceSize    stageCap   = 0;
    VkImage         stageImg   = VK_NULL_HANDLE;
    VkDeviceMemory  stageImgMem = VK_NULL_HANDLE;
    uint32_t        stageW = 0, stageH = 0;

    // --fps content sampling on the GPU present path (#3010). The swapchain blit reads the
    // renderer's image and never touches host memory, so the distinct-frame counter had nothing to
    // observe. These take a small NEAREST-filtered copy of the same image in the same command
    // buffer, which is a point-sample grid over the presented picture, and read it back one frame
    // late so no extra fence wait is introduced.
    //
    // Allocated only when --fps asked for the counter: this is an instrument, and an instrument that
    // runs when nobody asked for it is how a measurement pass ends up measuring itself (#2113).
    VkImage         sampleImg    = VK_NULL_HANDLE;
    VkDeviceMemory  sampleImgMem = VK_NULL_HANDLE;
    VkBuffer        sampleBuf    = VK_NULL_HANDLE;
    VkDeviceMemory  sampleMem    = VK_NULL_HANDLE;
    void*           sampleMapped = nullptr;
    // A sample recorded last present and not yet signed. The fence that guards it is inFlight,
    // which present_frame_gpu already waits on at entry, so the read is free.
    bool            samplePending = false;


    VkCommandPool   cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd     = VK_NULL_HANDLE;
    VkSemaphore     acquireSem = VK_NULL_HANDLE;
    // #2403: ONE present semaphore PER SWAPCHAIN IMAGE, indexed by the acquired image index.
    // A single shared semaphore is signalled by vkQueueSubmit while the presentation engine may still
    // hold it from the previous image's present -- VUID-vkQueueSubmit-pSignalSemaphores-00067, 6x per
    // GTA V run. The layer's own remedy (a); needs no extension.
    std::vector<VkSemaphore> presentSems;
    VkSemaphore     presentSem = VK_NULL_HANDLE;   // retained: fallback before the vector is sized
    VkFence         inFlight   = VK_NULL_HANDLE;

    // Present unification (#1270): set when the app adopted the renderer's shared device and presents by
    // GPU-blitting its front-buffer image (no CPU round-trip). queue_shared means the present queue aliases
    // the render queue, so present submits serialize through gpu::shared_present_submit_mutex().
    bool            gpu_present = false;
    bool            queue_shared = false;

    // --fps. Null unless the counter was asked for AND came up. When it is live it REPLACES the
    // present path's final TRANSFER_DST -> PRESENT_SRC barrier with its own render pass, so both
    // present paths ask it first and fall back to the barrier when it declines.
    prosper::frontend::FpsOverlay* overlay = nullptr;
};

uint32_t find_mem(VkPhysicalDevice p, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties m; vkGetPhysicalDeviceMemoryProperties(p, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (m.memoryTypes[i].propertyFlags & props) == props) return i;
    return UINT32_MAX;
}

bool create_instance(Vk& vk, SDL_Window* win) {
    uint32_t extCount = 0;
    const char* const* sdlExts = SDL_Vulkan_GetInstanceExtensions(&extCount);
    if (!sdlExts) { fprintf(stderr, "[app] SDL_Vulkan_GetInstanceExtensions: %s\n", SDL_GetError()); return false; }
    std::vector<const char*> exts(sdlExts, sdlExts + extCount);

    // macOS: SDL adds VK_KHR_portability_enumeration to its required extensions because it assumes
    // the Khronos loader. This build links MoltenVK DIRECTLY (no loader), where MoltenVK is the sole
    // driver — no enumeration is needed and MoltenVK rejects the extension with
    // VK_ERROR_EXTENSION_NOT_PRESENT. Strip it (and never set the enumerate flag). The device-level
    // VK_KHR_portability_subset (below) is the piece that actually matters. If a future build routes
    // through the loader, keep SDL's extension and add the ENUMERATE_PORTABILITY flag instead.
#ifdef __APPLE__
    exts.erase(std::remove_if(exts.begin(), exts.end(),
                   [](const char* e){ return std::strcmp(e, "VK_KHR_portability_enumeration") == 0; }),
               exts.end());
#endif
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    if (!prosper::frontend::require_vulkan_runtime_loader("app")) return false;
    app.pApplicationName = "prosper-app";
    app.apiVersion = prosper::frontend::kVulkanRuntimeVersion;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    VKCHECK(vkCreateInstance(&ci, nullptr, &vk.instance), "vkCreateInstance");

    if (!SDL_Vulkan_CreateSurface(win, vk.instance, nullptr, &vk.surface)) {
        fprintf(stderr, "[app] SDL_Vulkan_CreateSurface: %s\n", SDL_GetError()); return false;
    }
    return true;
}

bool pick_device(Vk& vk) {
    uint32_t n = 0; vkEnumeratePhysicalDevices(vk.instance, &n, nullptr);
    if (!n) { fprintf(stderr, "[app] no Vulkan device\n"); return false; }
    std::vector<VkPhysicalDevice> devs(n); vkEnumeratePhysicalDevices(vk.instance, &n, devs.data());
    for (auto d : devs) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(d, &properties);
        if (!prosper::frontend::vulkan_runtime_version_supported(properties.apiVersion)) continue;
        uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> q(qn); vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, q.data());
        for (uint32_t i = 0; i < qn; i++) {
            VkBool32 present = VK_FALSE; vkGetPhysicalDeviceSurfaceSupportKHR(d, i, vk.surface, &present);
            if ((q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) { vk.phys = d; vk.qfamily = i; break; }
        }
        if (vk.phys) break;
    }
    if (!vk.phys) { fprintf(stderr, "[app] no Vulkan 1.4 device with a graphics+present queue family\n"); return false; }

    float pr = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = vk.qfamily; qi.queueCount = 1; qi.pQueuePriorities = &pr;
    std::vector<const char*> devExts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
#ifdef __APPLE__
    // The Vulkan spec requires enabling VK_KHR_portability_subset on any device that advertises it
    // (all MoltenVK devices do), or vkCreateDevice fails with VK_ERROR_EXTENSION_NOT_PRESENT.
    { uint32_t n = 0; vkEnumerateDeviceExtensionProperties(vk.phys, nullptr, &n, nullptr);
      std::vector<VkExtensionProperties> dp(n);
      vkEnumerateDeviceExtensionProperties(vk.phys, nullptr, &n, dp.data());
      for (auto& e : dp) if (!strcmp(e.extensionName, "VK_KHR_portability_subset")) {
          devExts.push_back("VK_KHR_portability_subset"); break; } }
#endif
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    di.enabledExtensionCount = (uint32_t)devExts.size(); di.ppEnabledExtensionNames = devExts.data();
    VKCHECK(vkCreateDevice(vk.phys, &di, nullptr, &vk.device), "vkCreateDevice");
    vkGetDeviceQueue(vk.device, vk.qfamily, 0, &vk.queue);

    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(vk.phys, &pp);
    fprintf(stderr, "[app] Vulkan device: %s\n", pp.deviceName);
    prosper::frontend::log_vulkan_runtime_device("app", vk.phys, pp);
    return true;
}

bool create_swapchain(Vk& vk, uint32_t w, uint32_t h,
                      prosper::frontend::AppPresentMode requestedPresentMode) {
    VkSurfaceCapabilitiesKHR caps; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk.phys, vk.surface, &caps);
    vk.scExtent = prosper::frontend::select_swapchain_extent(caps.currentExtent, {w, h});
    if (vk.scExtent.width == 0 || vk.scExtent.height == 0) return false;   // minimized

    uint32_t fmtN = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(vk.phys, vk.surface, &fmtN, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtN); vkGetPhysicalDeviceSurfaceFormatsKHR(vk.phys, vk.surface, &fmtN, fmts.data());
    vk.scFormat = fmts[0].format; VkColorSpaceKHR cs = fmts[0].colorSpace;
    for (auto& f : fmts) if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { vk.scFormat = f.format; cs = f.colorSpace; break; }

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount && imgCount > caps.maxImageCount) imgCount = caps.maxImageCount;

    uint32_t modeN = 0;
    VKCHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(vk.phys, vk.surface, &modeN, nullptr),
            "vkGetPhysicalDeviceSurfacePresentModesKHR(count)");
    std::vector<VkPresentModeKHR> modes(modeN);
    VKCHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(vk.phys, vk.surface, &modeN, modes.data()),
            "vkGetPhysicalDeviceSurfacePresentModesKHR(list)");
    const bool hasMailbox = std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end();
    const bool hasImmediate = std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.end();
    const auto selectedPresentMode = prosper::frontend::select_present_mode(
        requestedPresentMode, hasMailbox, hasImmediate);
    VkPresentModeKHR vkPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (selectedPresentMode.mode == prosper::frontend::AppPresentMode::mailbox)
        vkPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    else if (selectedPresentMode.mode == prosper::frontend::AppPresentMode::immediate)
        vkPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    if (selectedPresentMode.fell_back) {
        fprintf(stderr, "[app] requested present mode %s is unsupported; falling back to fifo\n",
                prosper::frontend::present_mode_name(requestedPresentMode));
    }
    fprintf(stderr, "[app] present mode: %s\n",
            prosper::frontend::present_mode_name(selectedPresentMode.mode));

    VkSwapchainCreateInfoKHR si{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    si.surface = vk.surface; si.minImageCount = imgCount; si.imageFormat = vk.scFormat;
    si.imageColorSpace = cs; si.imageExtent = vk.scExtent; si.imageArrayLayers = 1;
    // TRANSFER_DST for the game path, which BLITS finished frames in; COLOR_ATTACHMENT so the library
    // view (#1471) can render into the same images through a real render pass. Without the attachment
    // bit its framebuffers are invalid and every draw is silently dropped — the render pass still
    // appears to clear, which makes it look like a UI bug rather than a swapchain one.
    si.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    si.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    si.preTransform = caps.currentTransform;
    si.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    si.presentMode = vkPresentMode;
    si.clipped = VK_TRUE;
    VKCHECK(vkCreateSwapchainKHR(vk.device, &si, nullptr, &vk.swapchain), "vkCreateSwapchainKHR");
    uint32_t n = 0; vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &n, nullptr);
    vk.scImages.resize(n); vkGetSwapchainImagesKHR(vk.device, vk.swapchain, &n, vk.scImages.data());
    return true;
}

// (Re)create the staging buffer + image sized to the guest frame (w*h RGBA).
bool ensure_stage(Vk& vk, uint32_t w, uint32_t h) {
    if (vk.stageW == w && vk.stageH == h && vk.stageBuf) return true;
    // The last presented frame may still be reading these objects, so a resize must not free them
    // under a submitted command buffer. Only the resize path pays the wait; the steady state above
    // returns before it. Reachable whenever the source dimensions change — a guest that
    // reconfigures VideoOut, or the idle window handing over to a booted game (#1469).
    if (vk.stageBuf || vk.stageImg) vkWaitForFences(vk.device, 1, &vk.inFlight, VK_TRUE, UINT64_MAX);
    if (vk.stageBuf)   { vkDestroyBuffer(vk.device, vk.stageBuf, nullptr); vkFreeMemory(vk.device, vk.stageMem, nullptr); }
    if (vk.stageImg)   { vkDestroyImage(vk.device, vk.stageImg, nullptr);  vkFreeMemory(vk.device, vk.stageImgMem, nullptr); }
    vk.stageBuf = VK_NULL_HANDLE; vk.stageImg = VK_NULL_HANDLE; vk.stageMapped = nullptr;
    vk.stageW = w; vk.stageH = h; vk.stageCap = (VkDeviceSize)w * h * 4;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = vk.stageCap;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKCHECK(vkCreateBuffer(vk.device, &bi, nullptr, &vk.stageBuf), "stage buffer");
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(vk.device, vk.stageBuf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_mem(vk.phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VKCHECK(vkAllocateMemory(vk.device, &ai, nullptr, &vk.stageMem), "stage buffer mem");
    vkBindBufferMemory(vk.device, vk.stageBuf, vk.stageMem, 0);
    vkMapMemory(vk.device, vk.stageMem, 0, vk.stageCap, 0, &vk.stageMapped);

    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {w, h, 1}; ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VKCHECK(vkCreateImage(vk.device, &ii, nullptr, &vk.stageImg), "stage image");
    vkGetImageMemoryRequirements(vk.device, vk.stageImg, &mr);
    ai.allocationSize = mr.size; ai.memoryTypeIndex = find_mem(vk.phys, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VKCHECK(vkAllocateMemory(vk.device, &ai, nullptr, &vk.stageImgMem), "stage image mem");
    vkBindImageMemory(vk.device, vk.stageImg, vk.stageImgMem, 0);
    return true;
}

// The --fps content-sample grid (#3010). Fixed size, so it survives swapchain and source-extent
// changes untouched: the signature only has to be comparable with the PREVIOUS signature, and a grid
// that resized would make consecutive frames incomparable for one frame every time the guest
// reconfigured VideoOut -- reporting a content change that did not happen.
//
// 256x144 is 36,864 points, above the ~8,100 pixels frame_content_signature samples from a 1080p
// frame, and it costs 147,456 bytes of host memory and one small blit per present.
constexpr uint32_t kFpsSampleW = 256, kFpsSampleH = 144;
constexpr VkDeviceSize kFpsSampleBytes = (VkDeviceSize)kFpsSampleW * kFpsSampleH * 4;

// True once the sample grid is usable. Failure is not fatal: --fps degrades to the unmeasured HUD
// rather than taking the app down over an instrument. It IS re-attempted, because the caller runs
// this whenever the overlay is not yet ready -- the early-out on the first line is what makes a
// success cheap, not a latch that would stop a retry.
bool ensure_fps_sample(Vk& vk) {
    if (vk.sampleBuf) return true;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = kFpsSampleBytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk.device, &bi, nullptr, &vk.sampleBuf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(vk.device, vk.sampleBuf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_mem(vk.phys, mr.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(vk.device, &ai, nullptr, &vk.sampleMem) != VK_SUCCESS) goto fail;
    vkBindBufferMemory(vk.device, vk.sampleBuf, vk.sampleMem, 0);
    if (vkMapMemory(vk.device, vk.sampleMem, 0, kFpsSampleBytes, 0, &vk.sampleMapped) != VK_SUCCESS)
        goto fail;

    {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R8G8B8A8_UNORM;
        ii.extent = {kFpsSampleW, kFpsSampleH, 1};
        ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(vk.device, &ii, nullptr, &vk.sampleImg) != VK_SUCCESS) goto fail;
        VkMemoryRequirements imr; vkGetImageMemoryRequirements(vk.device, vk.sampleImg, &imr);
        VkMemoryAllocateInfo iai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        iai.allocationSize = imr.size;
        iai.memoryTypeIndex = find_mem(vk.phys, imr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(vk.device, &iai, nullptr, &vk.sampleImgMem) != VK_SUCCESS) goto fail;
        vkBindImageMemory(vk.device, vk.sampleImg, vk.sampleImgMem, 0);
    }
    return true;

fail:
    fprintf(stderr, "[fps] content sampling unavailable; the HUD will report the presented rate only\n");
    if (vk.sampleMapped) { vkUnmapMemory(vk.device, vk.sampleMem); vk.sampleMapped = nullptr; }
    if (vk.sampleImg)    { vkDestroyImage(vk.device, vk.sampleImg, nullptr); vk.sampleImg = VK_NULL_HANDLE; }
    if (vk.sampleImgMem) { vkFreeMemory(vk.device, vk.sampleImgMem, nullptr); vk.sampleImgMem = VK_NULL_HANDLE; }
    if (vk.sampleBuf)    { vkDestroyBuffer(vk.device, vk.sampleBuf, nullptr); vk.sampleBuf = VK_NULL_HANDLE; }
    if (vk.sampleMem)    { vkFreeMemory(vk.device, vk.sampleMem, nullptr); vk.sampleMem = VK_NULL_HANDLE; }
    return false;
}

void barrier(VkCommandBuffer c, VkImage img, VkImageLayout from, VkImageLayout to,
             VkAccessFlags srcA, VkAccessFlags dstA, VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from; b.newLayout = to; b.image = img;
    b.srcAccessMask = srcA; b.dstAccessMask = dstA;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(c, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// A failed queue submit leaves inFlight unsignaled and the acquired-image semaphore unusable for a
// second acquire. Replace the whole sync trio before asking the main loop to rebuild the swapchain;
// otherwise the next frame would wait forever on a fence that no submit can signal. The abandoned
// handles are deliberately retained until process teardown: the acquire signal operation may still
// be pending, so destroying them here would itself violate Vulkan lifetime rules. This path is rare
// and bounded by a fatal device/driver error.
// #2403: the present semaphore for one acquired swapchain image, created on first use of that index.
// Lazy rather than sized up-front because the acquired-index domain is whatever the driver hands back;
// growing on demand needs no image-count plumbing and cannot under-size. Falls back to the single
// shared semaphore only if creation fails, which is exactly the old (incorrect but working) behaviour.
VkSemaphore present_sem_for(Vk& vk, uint32_t image_index) {
    if (image_index >= vk.presentSems.size()) vk.presentSems.resize(image_index + 1, VK_NULL_HANDLE);
    if (vk.presentSems[image_index] == VK_NULL_HANDLE) {
        VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore s = VK_NULL_HANDLE;
        if (vkCreateSemaphore(vk.device, &semi, nullptr, &s) == VK_SUCCESS) vk.presentSems[image_index] = s;
        else return vk.presentSem;
    }
    return vk.presentSems[image_index];
}

bool replace_present_sync(Vk& vk) {
    VkSemaphore acquire = VK_NULL_HANDLE;
    VkSemaphore present = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateSemaphore(vk.device, &semi, nullptr, &acquire) != VK_SUCCESS ||
        vkCreateSemaphore(vk.device, &semi, nullptr, &present) != VK_SUCCESS ||
        vkCreateFence(vk.device, &fci, nullptr, &fence) != VK_SUCCESS) {
        if (fence) vkDestroyFence(vk.device, fence, nullptr);
        if (present) vkDestroySemaphore(vk.device, present, nullptr);
        if (acquire) vkDestroySemaphore(vk.device, acquire, nullptr);
        return false;
    }
    vk.acquireSem = acquire;
    vk.presentSem = present;
    // #2403: drop the per-image set; present_sem_for() re-creates lazily per acquired index.
    for (VkSemaphore s : vk.presentSems)
        if (s) vkDestroySemaphore(vk.device, s, nullptr);
    vk.presentSems.clear();
    vk.inFlight = fence;
    return true;
}

prosper::frontend::PresentAttempt recover_submit_failure(Vk& vk, VkResult result) {
    if (prosper::frontend::classify_submit_failure(result) ==
        prosper::frontend::PresentAttempt::failed) {
        fprintf(stderr, "[app] vkQueueSubmit failed (%d); device lost, stopping\n",
                static_cast<int>(result));
        return prosper::frontend::PresentAttempt::failed;
    }
    fprintf(stderr, "[app] vkQueueSubmit failed (%d); replacing present synchronization\n",
            static_cast<int>(result));
    if (replace_present_sync(vk)) return prosper::frontend::PresentAttempt::out_of_date;
    fprintf(stderr, "[app] could not replace present synchronization; stopping\n");
    return prosper::frontend::PresentAttempt::failed;
}

// Interactive frame grab (F9): write a small BGR bottom-up BMP of a presented RGBA frame next to the
// .prgcap so the user gets a visible "this is the frame I captured" alongside the replayable capsule.
// A local writer (the shared dump_bmp lives in the Vulkan test harness, not linkable here) — the .prgcap
// oracle remains the authoritative pixels; this is the convenience screenshot.
static bool write_frame_bmp(const std::string& path, const uint8_t* rgba, uint32_t w, uint32_t h) {
    if (!rgba || !w || !h) return false;
    const uint32_t row = w * 3, pad = (4 - (row & 3)) & 3, stride = row + pad;
    const uint32_t pixels = stride * h, size = 54 + pixels;
    std::vector<uint8_t> f(size, 0);
    auto put16 = [&](uint32_t o, uint16_t v) { f[o] = v & 0xff; f[o + 1] = v >> 8; };
    auto put32 = [&](uint32_t o, uint32_t v) { for (int i = 0; i < 4; i++) f[o + i] = (v >> (8 * i)) & 0xff; };
    f[0] = 'B'; f[1] = 'M'; put32(2, size); put32(10, 54);
    put32(14, 40); put32(18, w); put32(22, h); put16(26, 1); put16(28, 24); put32(34, pixels);
    for (uint32_t y = 0; y < h; y++) {
        uint8_t* dst = f.data() + 54 + (size_t)(h - 1 - y) * stride;   // BMP rows are bottom-up
        const uint8_t* src = rgba + (size_t)y * w * 4;
        for (uint32_t x = 0; x < w; x++) { dst[x * 3] = src[x * 4 + 2]; dst[x * 3 + 1] = src[x * 4 + 1]; dst[x * 3 + 2] = src[x * 4]; }
    }
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) return false;
    const bool ok = fwrite(f.data(), 1, f.size(), fp) == f.size();
    fclose(fp);
    return ok;
}

// Present one guest RGBA frame (w*h, 4 bytes/pixel) to the window, scaling to the swapchain extent.
using prosper::frontend::PresentAttempt;

prosper::frontend::PresentAttempt present_frame(Vk& vk, const uint8_t* rgba, uint32_t w, uint32_t h,
                                                const std::vector<std::string>* overlayLines = nullptr) {
    if (!ensure_stage(vk, w, h)) return PresentAttempt::out_of_date;
    memcpy(vk.stageMapped, rgba, (size_t)w * h * 4);

    const VkResult waitResult = vkWaitForFences(
        vk.device, 1, &vk.inFlight, VK_TRUE, UINT64_MAX);
    if (waitResult != VK_SUCCESS) {
        fprintf(stderr, "[app] present fence wait failed (%d); stopping\n",
                static_cast<int>(waitResult));
        return PresentAttempt::failed;
    }
    uint32_t imgIndex = 0;
    // Bound the acquire (#1182): an occluded/minimized window releases no swapchain image, and an
    // infinite wait here would block the app main thread — freezing visible output, stalling SDL
    // event/close handling, and (on the shared physical GPU) potentially the guest. A timeout is a
    // benign SKIP: return without touching the swapchain and retry next loop. The guest keeps running
    // on its own device regardless.
    constexpr uint64_t kAcquireTimeoutNs = 100ull * 1000 * 1000;   // 100 ms — never hit while visible
    VkResult acq = vkAcquireNextImageKHR(vk.device, vk.swapchain, kAcquireTimeoutNs, vk.acquireSem,
                                         VK_NULL_HANDLE, &imgIndex);
    switch (prosper::frontend::classify_acquire(acq)) {
    case prosper::frontend::AcquireAction::skip:     return PresentAttempt::skipped;
    case prosper::frontend::AcquireAction::recreate: return PresentAttempt::out_of_date;
    case prosper::frontend::AcquireAction::fail:     return PresentAttempt::failed;
    case prosper::frontend::AcquireAction::proceed:  break;
    }
    // Load-bearing ordering: vkResetFences MUST stay after the skip/recreate early-returns above. A skip
    // leaves inFlight signaled (from the last real present) so the next frame's vkWaitForFences returns
    // immediately; hoisting this reset above the acquire would leave inFlight unsignaled on a skip with no
    // paired submit to re-signal it, and the next vkWaitForFences would hang. The reset is always paired
    // with the vkQueueSubmit(..., inFlight) below.
    vkResetFences(vk.device, 1, &vk.inFlight);
    vkResetCommandBuffer(vk.cmd, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(vk.cmd, &bi);

    // staging buffer -> staging image (TRANSFER_DST), then image -> swapchain image (blit, scaled).
    barrier(vk.cmd, vk.stageImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy bic{}; bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; bic.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(vk.cmd, vk.stageBuf, vk.stageImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
    barrier(vk.cmd, vk.stageImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    barrier(vk.cmd, vk.scImages[imgIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; blit.srcOffsets[1] = {(int32_t)w, (int32_t)h, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {(int32_t)vk.scExtent.width, (int32_t)vk.scExtent.height, 1};
    vkCmdBlitImage(vk.cmd, vk.stageImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   vk.scImages[imgIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    // The overlay's render pass loads the blitted contents and leaves the image in PRESENT_SRC, so
    // it stands in for the barrier below rather than adding to it. With --fps off, or if the HUD
    // declined, this is byte-for-byte the path it always was.
    if (!(overlayLines && vk.overlay && vk.overlay->record(vk.cmd, imgIndex, *overlayLines)))
        barrier(vk.cmd, vk.scImages[imgIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    vkEndCommandBuffer(vk.cmd);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo su{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    su.waitSemaphoreCount = 1; su.pWaitSemaphores = &vk.acquireSem; su.pWaitDstStageMask = &waitStage;
    su.commandBufferCount = 1; su.pCommandBuffers = &vk.cmd;
    // #2403: bind the per-image present semaphore for THIS acquired index before use.
    // Signalled by the submit below and waited by the present; sharing one across images
    // is VUID-vkQueueSubmit-pSignalSemaphores-00067 (6x/run on GTA V).
    vk.presentSem = present_sem_for(vk, imgIndex);
    su.signalSemaphoreCount = 1; su.pSignalSemaphores = &vk.presentSem;

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &vk.presentSem;
    pi.swapchainCount = 1; pi.pSwapchains = &vk.swapchain; pi.pImageIndices = &imgIndex;
    VkResult submitResult;
    VkResult pr = VK_SUCCESS;
    {
        // #1270: when this CPU present runs on the renderer's shared queue (the GPU-present miss fallback),
        // serialize the submit + present CALLS against the renderer's submits. No-op on a private device.
        std::unique_lock<std::mutex> lk(gpu::shared_present_submit_mutex(), std::defer_lock);
        if (gpu::shared_present_active()) lk.lock();
        submitResult = vkQueueSubmit(vk.queue, 1, &su, vk.inFlight);
        if (submitResult == VK_SUCCESS) pr = vkQueuePresentKHR(vk.queue, &pi);
    }
    if (submitResult != VK_SUCCESS) return recover_submit_failure(vk, submitResult);
    return prosper::frontend::classify_present(pr);
}

// Present unification (#1270): try to present on the RENDERER's Vulkan device instead of a private one,
// so the app can GPU-blit the renderer's front-buffer image straight to the swapchain (no 4K CPU
// round-trip). Returns false at any unmet precondition, leaving the caller to create its own device (the
// unchanged CPU path). Called only when PROSPER_APP_GPU_PRESENT is set and a game is booting.
bool try_adopt_shared_present(Vk& vk, SDL_Window* win) {
    const gpu::SharedVulkanContext ctx = gpu::shared_vulkan_context();
    if (!ctx.valid() || !ctx.present_capable || !ctx.present_queue) {
        fprintf(stderr, "[app] GPU present: shared device is not present-capable; using own device\n");
        return false;
    }
    VkInstance inst = (VkInstance)ctx.instance;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(win, inst, nullptr, &surface)) {
        // The shared instance was created before SDL_Init and enabled every AVAILABLE platform surface
        // extension blind; if SDL needs one we did not get, fall back to a private device.
        fprintf(stderr, "[app] GPU present: surface on shared instance failed (%s); using own device\n",
                SDL_GetError());
        return false;
    }
    VkPhysicalDevice phys = (VkPhysicalDevice)ctx.physical;
    VkBool32 present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(phys, ctx.queue_family, surface, &present);
    if (!present) {
        fprintf(stderr, "[app] GPU present: shared queue family cannot present here; using own device\n");
        vkDestroySurfaceKHR(inst, surface, nullptr);
        return false;
    }
    vk.instance = inst; vk.surface = surface; vk.phys = phys;
    vk.device = (VkDevice)ctx.device; vk.qfamily = ctx.queue_family;
    vk.queue = (VkQueue)ctx.present_queue;
    vk.gpu_present = true;
    vk.queue_shared = ctx.present_queue_shared;
    // The renderer must now publish the front-buffer image (and skip the CPU readback); if the present
    // queue aliases the render queue, both threads must serialize their submits.
    gpu::set_gpu_present_active(true);
    if (ctx.present_queue_shared) gpu::set_shared_present_active(true);
    fprintf(stderr, "[app] GPU present: adopted the renderer's device (%s present queue)\n",
            ctx.present_queue_shared ? "shared" : "dedicated");
    return true;
}

// Present one GPU scanout frame (the renderer's front-buffer image, already in TRANSFER_SRC_OPTIMAL) by
// blitting it straight to the swapchain -- no CPU pixels. `prevSlot` is the slot presented last frame; its
// GPU read is complete once inFlight signals below, so it is released then. Updates prevSlot to the slot
// now in flight and returns the acquire/present outcome.
prosper::frontend::PresentAttempt present_frame_gpu(Vk& vk, const prosper::frontend::GpuScanoutFrame& gf,
                                                    int& prevSlot, bool requestReadback,
                                                    bool& readbackReady, bool sampleContent,
                                                    const std::vector<std::string>* overlayLines = nullptr) {
    readbackReady = false;
    const VkResult previousWait = vkWaitForFences(
        vk.device, 1, &vk.inFlight, VK_TRUE, UINT64_MAX);   // previous present's read complete
    if (previousWait != VK_SUCCESS) {
        prosper::frontend::present_blit_release(gf.slot);
        if (prevSlot >= 0) {
            prosper::frontend::present_blit_release(prevSlot);
            prevSlot = -1;
        }
        fprintf(stderr, "[app] gpu-present fence wait failed (%d); stopping\n",
                static_cast<int>(previousWait));
        return prosper::frontend::PresentAttempt::failed;
    }
    // The wait above is the same fence that guarded last present's sample copy, so the sample is
    // readable here at no cost. Signing it one frame late is invisible in a rate: the counter needs
    // the interval between distinct frames, not the frame's own timestamp.
    if (vk.samplePending) {
        vk.samplePending = false;
        gpu::note_present_publication_signature(
            gpu::dense_content_signature(static_cast<const uint8_t*>(vk.sampleMapped),
                                         (size_t)kFpsSampleBytes));
    }
    if (prevSlot >= 0) { prosper::frontend::present_blit_release(prevSlot); prevSlot = -1; }
    if (requestReadback && !ensure_stage(vk, gf.width, gf.height)) {
        prosper::frontend::present_blit_release(gf.slot);
        fprintf(stderr, "[grab] could not allocate the gpu-present readback buffer\n");
        return prosper::frontend::PresentAttempt::failed;
    }

    uint32_t imgIndex = 0;
    constexpr uint64_t kAcquireTimeoutNs = 100ull * 1000 * 1000;
    VkResult acq = vkAcquireNextImageKHR(vk.device, vk.swapchain, kAcquireTimeoutNs, vk.acquireSem,
                                         VK_NULL_HANDLE, &imgIndex);
    switch (prosper::frontend::classify_acquire(acq)) {
    case prosper::frontend::AcquireAction::skip:
        prosper::frontend::present_blit_release(gf.slot); return prosper::frontend::PresentAttempt::skipped;
    case prosper::frontend::AcquireAction::recreate:
        prosper::frontend::present_blit_release(gf.slot); return prosper::frontend::PresentAttempt::out_of_date;
    case prosper::frontend::AcquireAction::fail:
        prosper::frontend::present_blit_release(gf.slot); return prosper::frontend::PresentAttempt::failed;
    case prosper::frontend::AcquireAction::proceed: break;
    }
    vkResetFences(vk.device, 1, &vk.inFlight);
    vkResetCommandBuffer(vk.cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(vk.cmd, &bi);
    // gf.image is already TRANSFER_SRC_OPTIMAL (left there by present_blit); swapchain image
    // UNDEFINED -> TRANSFER_DST -> (scaled blit) -> PRESENT_SRC.
    barrier(vk.cmd, vk.scImages[imgIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {(int32_t)gf.width, (int32_t)gf.height, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {(int32_t)vk.scExtent.width, (int32_t)vk.scExtent.height, 1};
    vkCmdBlitImage(vk.cmd, gf.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   vk.scImages[imgIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    // --fps content sample (#3010): a NEAREST-filtered reduction of the SAME image the swapchain
    // blit just read, copied to host memory. NEAREST rather than LINEAR on purpose -- averaging
    // would let a small bright change cancel against its neighbours, and this exists to detect
    // change, not to look good. Signed at the top of the NEXT present, under the fence already
    // waited on there, so this adds no synchronisation.
    if (sampleContent && vk.sampleImg) {
        barrier(vk.cmd, vk.sampleImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkImageBlit sb{};
        sb.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        sb.srcOffsets[1] = {(int32_t)gf.width, (int32_t)gf.height, 1};
        sb.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        sb.dstOffsets[1] = {(int32_t)kFpsSampleW, (int32_t)kFpsSampleH, 1};
        vkCmdBlitImage(vk.cmd, gf.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       vk.sampleImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &sb, VK_FILTER_NEAREST);
        barrier(vk.cmd, vk.sampleImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy sc{};
        sc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        sc.imageExtent = {kFpsSampleW, kFpsSampleH, 1};
        vkCmdCopyImageToBuffer(vk.cmd, vk.sampleImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               vk.sampleBuf, 1, &sc);
        VkBufferMemoryBarrier sh{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        sh.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sh.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        sh.srcQueueFamilyIndex = sh.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sh.buffer = vk.sampleBuf; sh.offset = 0; sh.size = kFpsSampleBytes;
        vkCmdPipelineBarrier(vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &sh, 0, nullptr);
        vk.samplePending = true;
    }

    if (requestReadback) {
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {gf.width, gf.height, 1};
        vkCmdCopyImageToBuffer(vk.cmd, gf.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               vk.stageBuf, 1, &copy);
        VkBufferMemoryBarrier hostBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        hostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hostBarrier.buffer = vk.stageBuf;
        hostBarrier.offset = 0;
        hostBarrier.size = vk.stageCap;
        vkCmdPipelineBarrier(vk.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &hostBarrier,
                             0, nullptr);
    }
    // Same substitution as the CPU path. Note the readback above copies gf.image -- the RENDERER's
    // frame -- so an F9 grab or a scheduled screenshot never carries the HUD.
    if (!(overlayLines && vk.overlay && vk.overlay->record(vk.cmd, imgIndex, *overlayLines)))
        barrier(vk.cmd, vk.scImages[imgIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    vkEndCommandBuffer(vk.cmd);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo su{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    su.waitSemaphoreCount = 1; su.pWaitSemaphores = &vk.acquireSem; su.pWaitDstStageMask = &waitStage;
    su.commandBufferCount = 1; su.pCommandBuffers = &vk.cmd;
    // #2403: bind the per-image present semaphore for THIS acquired index before use.
    // Signalled by the submit below and waited by the present; sharing one across images
    // is VUID-vkQueueSubmit-pSignalSemaphores-00067 (6x/run on GTA V).
    vk.presentSem = present_sem_for(vk, imgIndex);
    su.signalSemaphoreCount = 1; su.pSignalSemaphores = &vk.presentSem;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &vk.presentSem;
    pi.swapchainCount = 1; pi.pSwapchains = &vk.swapchain; pi.pImageIndices = &imgIndex;
    VkResult submitResult;
    VkResult pr = VK_SUCCESS;
    {
        // Serialize the present submit + present CALL against the renderer's submits on a shared queue.
        std::unique_lock<std::mutex> lk(gpu::shared_present_submit_mutex(), std::defer_lock);
        if (vk.queue_shared) lk.lock();
        submitResult = vkQueueSubmit(vk.queue, 1, &su, vk.inFlight);
        if (submitResult == VK_SUCCESS) pr = vkQueuePresentKHR(vk.queue, &pi);
    }
    if (submitResult != VK_SUCCESS) {
        prosper::frontend::present_blit_release(gf.slot);
        // The sample copy was RECORDED but never executed, so the buffer still holds the previous
        // frame (or is indeterminate on the first pass). Clearing the flag keeps the counter from
        // signing bytes no present produced -- one spurious distinct-or-not verdict, in bounds and
        // harmless, but it would be a measurement of nothing.
        vk.samplePending = false;
        return recover_submit_failure(vk, submitResult);
    }

    prevSlot = gf.slot;
    if (requestReadback) {
        const VkResult waitResult = vkWaitForFences(vk.device, 1, &vk.inFlight, VK_TRUE, UINT64_MAX);
        if (waitResult != VK_SUCCESS) {
            fprintf(stderr, "[grab] gpu-present readback wait failed (%d)\n",
                    static_cast<int>(waitResult));
            return prosper::frontend::PresentAttempt::failed;
        }
        prosper::frontend::present_blit_release(prevSlot);
        prevSlot = -1;
        readbackReady = true;
    }
    return prosper::frontend::classify_present(pr);
}

// Synthetic animated frame (no guest): a moving gradient fed through the REAL present layer, so the
// window + swapchain + readback path is exercised end-to-end without a game dump.
void feed_test_pattern(uint32_t w, uint32_t h, uint64_t frame) {
    static std::vector<uint8_t> px; px.resize((size_t)w * h * 4);
    uint8_t t = (uint8_t)(frame * 2);
    for (uint32_t y = 0; y < h; y++) for (uint32_t x = 0; x < w; x++) {
        uint8_t* p = &px[((size_t)y * w + x) * 4];
        p[0] = (uint8_t)(x * 255 / w + t);   // R
        p[1] = (uint8_t)(y * 255 / h);       // G
        p[2] = t;                            // B
        p[3] = 255;
    }
    gpu::present_write_frame(px.data(), w, h);
}

// ---- keyboard -> virtual DualSense (pad 0) ----------------------------------------------------
// A controller over WSL passthrough is flaky, so the app maps the keyboard onto the same
// HostPadState the SDL gamepad backend fills. The event loop updates g_keyboard_pad on the main
// thread; the guest's scePadReadState reads the composed keyboard/physical state on a guest thread.
prosper::frontend::KeyboardPadOverlay g_keyboard_pad;

// The only SDL-aware part of the keyboard map: which physical key each PadKey names. The mapping
// itself is in keyboard_pad_map.hpp, where a test can reach it without linking SDL.
static SDL_Scancode scancode_for(prosper::frontend::PadKey k) {
    using prosper::frontend::PadKey;
    switch (k) {
        case PadKey::W:          return SDL_SCANCODE_W;
        case PadKey::A:          return SDL_SCANCODE_A;
        case PadKey::S:          return SDL_SCANCODE_S;
        case PadKey::D:          return SDL_SCANCODE_D;
        case PadKey::T:          return SDL_SCANCODE_T;
        case PadKey::F:          return SDL_SCANCODE_F;
        case PadKey::G:          return SDL_SCANCODE_G;
        case PadKey::H:          return SDL_SCANCODE_H;
        case PadKey::I:          return SDL_SCANCODE_I;
        case PadKey::J:          return SDL_SCANCODE_J;
        case PadKey::K:          return SDL_SCANCODE_K;
        case PadKey::L:          return SDL_SCANCODE_L;
        case PadKey::N:          return SDL_SCANCODE_N;
        case PadKey::M:          return SDL_SCANCODE_M;
        case PadKey::Comma:      return SDL_SCANCODE_COMMA;
        case PadKey::Period:     return SDL_SCANCODE_PERIOD;
        case PadKey::Z:          return SDL_SCANCODE_Z;
        case PadKey::X:          return SDL_SCANCODE_X;
        case PadKey::C:          return SDL_SCANCODE_C;
        case PadKey::V:          return SDL_SCANCODE_V;
        case PadKey::B:          return SDL_SCANCODE_B;
        case PadKey::Slash:      return SDL_SCANCODE_SLASH;
        case PadKey::Space:      return SDL_SCANCODE_SPACE;
        case PadKey::ArrowUp:    return SDL_SCANCODE_UP;
        case PadKey::ArrowDown:  return SDL_SCANCODE_DOWN;
        case PadKey::ArrowLeft:  return SDL_SCANCODE_LEFT;
        case PadKey::ArrowRight: return SDL_SCANCODE_RIGHT;
        // Enter has three scancodes and is handled by the caller; Count is not a key.
        case PadKey::Enter:
        case PadKey::Count:      break;
    }
    return SDL_SCANCODE_UNKNOWN;
}

// Snapshot the current keyboard into the overlay. Call from the thread that pumps SDL events.
void poll_keyboard(const bool* keyboard, bool enter_maps_to_options) {
    using prosper::frontend::PadKey;
    const auto down = [&](PadKey k) {
        // Return, the numeric keypad's Enter, and the ISO Return2 are all "Enter" to a player.
        if (k == PadKey::Enter)
            return keyboard[SDL_SCANCODE_RETURN] || keyboard[SDL_SCANCODE_RETURN2] ||
                   keyboard[SDL_SCANCODE_KP_ENTER];
        const SDL_Scancode s = scancode_for(k);
        return s != SDL_SCANCODE_UNKNOWN && keyboard[s];
    };
    g_keyboard_pad.set_keyboard_state(
        prosper::frontend::map_keyboard_to_pad(down, enter_maps_to_options));
}

// ---- opening a game (#1469) --------------------------------------------------------------------
// A dump path reaches the boot from three places: argv (the primary, agentic path — unchanged), a
// folder dropped on the window, and the host folder picker. Only the first is available before the
// window exists, so the boot itself is factored out here and every source runs the identical code.

// The host side of game_path.hpp's injected probe. Both lambdas go through resolve_host_path_case()
// for the same reason boot_program does (#1006/#1226): the PS5 filesystem namespace is
// case-insensitive and titles disagree on casing, so an exact-case probe on a case-sensitive host
// would reject a dump that boot_program would then have loaded perfectly well.
static prosper::frontend::GamePathProbe host_path_probe() {
    prosper::frontend::GamePathProbe probe;
    probe.is_dir = [](const std::string& p) {
        struct stat st{};
        return ::stat(resolve_host_path_case(p).c_str(), &st) == 0 && (st.st_mode & S_IFMT) == S_IFDIR;
    };
    probe.is_file = [](const std::string& p) {
        struct stat st{};
        return ::stat(resolve_host_path_case(p).c_str(), &st) == 0 && (st.st_mode & S_IFMT) == S_IFREG;
    };
    return probe;
}

// Read a file whole, exactly as named. Returns "" when it cannot be opened.
static std::string read_whole_file(const std::string& path) {
    std::string out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    char buf[8192]; size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

// The host side of game_library.hpp's injected IO.
static prosper::frontend::GameLibraryIo host_library_io() {
    prosper::frontend::GameLibraryIo io;
    io.list_dir = [](const std::string& dir) {
        std::vector<std::string> names;
        // Both the construction AND the increment take an error_code: the throwing operator++ is
        // reachable when a directory is removed or becomes unreadable mid-scan, and that must not
        // escape as an exception out of an ordinary listing.
        std::error_code ec;
        std::filesystem::directory_iterator it(dir, ec);
        const std::filesystem::directory_iterator end;
        while (!ec && it != end) {
            names.push_back(it->path().filename().string());
            it.increment(ec);
        }
        // A partial listing is the right recovery, but showing fewer games with no explanation is not.
        if (ec) fprintf(stderr, "[app] listing %s stopped early: %s\n", dir.c_str(), ec.message().c_str());
        return names;
    };
    io.read_file = [](const std::string& want) {
        // Case-correct like boot_program does (#1006/#1226): a dump shipping SCE_SYS/PARAM.JSON on a
        // case-sensitive host would otherwise pass the probe and then read as empty, silently
        // demoting the title to its directory name. GUEST paths only — see read_whole_file.
        return read_whole_file(resolve_host_path_case(want));
    };
    io.resolve_case = [](const std::string& want) { return resolve_host_path_case(want); };
    return io;
}

// Read the human-readable game name from the dump's PS5 param.json so the window title shows e.g.
// "Bendy and the Ink Machine" instead of the PPSA content-id directory. PS5 param.json stores the name under
// localizedParameters.<lang>.titleName; we prefer the defaultLanguage's entry and fall back to the first
// titleName found. Dependency-free string scan (no JSON lib in-tree); returns "" if the file/field is absent.
// Flips since the guest's first pad poll, or -1 before that poll happens. Read-only by contract:
// it never establishes the origin. Defined in src/hle/input/hle_pad.cpp.
extern "C" int64_t prosper_pad_flip_ordinal();

static std::string read_game_title(const std::string& dump) {
    // The parse lives in game_library.hpp so the library view and this window title read param.json
    // through exactly one implementation — and so that logic is unit-tested, which it never was while
    // it lived here (#1471).
    return prosper::frontend::parse_param_title_name(
        host_library_io().read_file(dump + "/sce_sys/param.json"));
}

// Where the persisted settings live. PROSPER_APP_CONFIG overrides it outright; otherwise the
// platform's per-user config location. Same shape as the project's other host paths
// (PROSPER_SAVEDATA_DIR, PROSPER_CAPTURE_DIR): an env override in front of a sensible default.
static std::string app_config_path() {
    if (const char* e = getenv("PROSPER_APP_CONFIG")) return e;
#ifdef _WIN32
    if (const char* appdata = getenv("APPDATA")) return std::string(appdata) + "\\prosper\\prosper-app.conf";
#else
    if (const char* xdg = getenv("XDG_CONFIG_HOME")) return std::string(xdg) + "/prosper/prosper-app.conf";
    if (const char* home = getenv("HOME")) return std::string(home) + "/.config/prosper/prosper-app.conf";
#endif
    return "";   // nowhere sensible to put it: run without persistence rather than guessing
}

static prosper::frontend::AppConfig load_app_config() {
    const std::string path = app_config_path();
    if (path.empty()) return {};
    // Read exactly, NOT through the case-correcting guest-path reader: this file's spelling is ours,
    // and case-correcting a path that usually does not exist scans its ancestor directories on every
    // first launch for no benefit.
    return prosper::frontend::parse_app_config(read_whole_file(path));
}

// Persist the settings. Best-effort: failing to write is reported and otherwise ignored, since the
// app is perfectly usable without persistence and a read-only home must not stop a game running.
static bool save_app_config(const prosper::frontend::AppConfig& cfg) {
    const std::string path = app_config_path();
    if (path.empty()) return false;
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    const std::string text = prosper::frontend::serialize_app_config(cfg);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "[app] could not write settings to %s\n", path.c_str()); return false; }
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    std::fclose(f);
    if (!ok) fprintf(stderr, "[app] settings write to %s was incomplete\n", path.c_str());
    return ok;
}

// The title component of an F9 capture's filename, plus the label its log line says.
//
// The FILENAME uses the content id ("PPSA25009") and never the display name: it is short, ASCII,
// locale-independent, free of spaces and punctuation, and it is already how this project identifies a
// title everywhere else (PROSPER_CAPTURE_TITLE, the game: issue labels, the status docs). Display
// names are the opposite on all four counts — PPSA20052's is "Worms Armageddon: Anniversary Edition"
// and PPSA08576's is "Asterix & Obelix Slap Them All!", and a language lookup that goes wrong yields
// "DER KÜHNE KNAPPE" for The Plucky Squire (#1471). The display name still appears in the
// arming LOG line, where none of that matters and it helps a person recognise the run.
struct CaptureTitle { std::string id, label; };
static CaptureTitle capture_title_for(const std::string& dump) {
    CaptureTitle out;
    if (dump.empty()) return out;
    const std::string json = host_library_io().read_file(dump + "/sce_sys/param.json");
    out.id = prosper::frontend::sanitize_capture_component(
        prosper::frontend::parse_param_title_id(json));
    // param.json missing, unreadable, or without titleId: the dump directory is named after the
    // content id in every dump we have, so it is a faithful fallback rather than a guess.
    if (out.id.empty()) out.id = prosper::frontend::title_id_from_app0_path(dump);
    const std::string name = prosper::frontend::parse_param_title_name(json);
    out.label = out.id.empty() ? name : (name.empty() ? out.id : out.id + " (" + name + ")");
    return out;
}

// "prosper - <game name>" for a booted game (name from param.json, falling back to the app0
// basename), else a label that says what the empty window is waiting for.
static std::string window_title_for(const std::string& dump, bool test_pattern) {
    if (!dump.empty()) {
        std::string name = read_game_title(dump);
        if (name.empty()) {
            const auto sl = dump.find_last_of("/\\");
            name = (sl == std::string::npos ? dump : dump.substr(sl + 1));
        }
        return "prosper - " + name;
    }
    if (test_pattern) return "prosper - test pattern";
    return "prosper - no game (drop a game folder here, or press Ctrl+O)";
}

// The guest, and the one boot this process gets. run_entry() never observes prosper_request_stop(),
// so a booted guest cannot be torn down (#352) — hence the relaunch below rather than a second call
// to start_guest().
//
// The two flags are deliberately distinct. `g_guest_started` says a guest is live (the idle painter
// and the test-pattern feeder stop, the window title changes). `g_boot_attempted` says this
// process's single boot has been SPENT — it latches before boot_program() runs and is never
// cleared, because a failed attempt is just as unrepeatable as a successful one: boot_program
// appends into g_prog and re-runs one-shot global setup, so a second call would link the new title
// behind the failed one's modules and leave imgs[0] — what the guest thread enters — stale.
// Every open decision uses g_boot_attempted.
Program g_prog;
std::thread g_guest_thread;
bool g_guest_started = false;
// True when THIS process authored PROSPER_GUEST_ARGS from the settings file (start_guest). A
// relaunch must strip an app-authored value so the next title re-resolves its own (#2973 review).
bool g_guest_args_app_set = false;
bool g_boot_attempted = false;

// Install the host frontends (audio out, controller in, dialogs) at the same point boot_trace does:
// right after the built-in HLE is registered, before the guest runs. Each is built in only when its
// SDL3 frontend is enabled; a window app wants them all on by default.
static void install_host_backends() {
#ifdef PROSPER_AUDIO_FFMPEG
    if (prosper::ajm::install_ffmpeg_decoder_backend())
        fprintf(stderr, "[app] FFmpeg AJM audio decoder installed.\n");
    else
        fprintf(stderr, "[app] FFmpeg AJM audio decoder unavailable.\n");
#endif
#ifdef PROSPER_VIDEO_MF
    if (!getenv("PROSPER_APP_DISABLE_VIDEO")) {
        if (prosper::video::install_media_foundation_backend())
            fprintf(stderr, "[app] Media Foundation video backend installed.\n");
        else
            fprintf(stderr, "[app] Media Foundation video backend unavailable.\n");
    } else {
        fprintf(stderr, "[app] native video backend disabled.\n");
    }
#endif
#ifdef PROSPER_VIDEO_VAAPI
    if (!getenv("PROSPER_APP_DISABLE_VIDEO")) {
        if (prosper::video::install_vaapi_backend())
            fprintf(stderr, "[app] FFmpeg/VA-API video backend installed.\n");
        else
            fprintf(stderr, "[app] FFmpeg/VA-API video backend unavailable.\n");
    } else {
        fprintf(stderr, "[app] native video backend disabled.\n");
    }
#endif
    // Loud by design. Built without SDL3 audio support this whole block vanishes, prosper-app has
    // NO playback device, and nothing says so. The PCM is still decoded, mixed and delivered -- to
    // an internal sink that reports "open" while nothing plays, which is indistinguishable from a
    // title simply being silent. That cost a full investigation on GTA V (#2402).
#ifdef PROSPER_AUDIO_SDL3
    if (!getenv("PROSPER_APP_DISABLE_AUDIO")) {
        prosper::install_sdl3_audio_sink();
        prosper::set_sdl3_audio_gain(g_volume_percent / 100.0f);
        fprintf(stderr, "[app] audio volume %d%%%s\n", g_volume_percent,
                g_volume_percent == kDefaultVolumePercent ? " (default; --volume N to change)" : "");
    } else {
        fprintf(stderr, "[app] SDL audio backend disabled; using the realtime silent sink.\n");
    }
#else
    fprintf(stderr, "[app] BUILT WITHOUT SDL3 AUDIO SUPPORT -- there will be no sound. "
                    "Reconfigure with -DPROSPER_AUDIO_SDL3=ON.\n");
#endif
#ifdef PROSPER_PAD_SDL3
    if (!getenv("PROSPER_APP_DISABLE_PAD")) {
        if (prosper::install_sdl3_pad_backend()) {
            g_keyboard_pad.set_fallback(prosper::input::pad_backend());
            fprintf(stderr, "[app] controller backend installed.\n");
        }
    } else {
        fprintf(stderr, "[app] SDL controller backend disabled; keyboard and scripted input remain available.\n");
    }
#endif
#ifdef PROSPER_HAVE_DIALOG_SDL3
    if (!getenv("PROSPER_APP_DISABLE_DIALOG")) {
        prosper::install_sdl3_platform_ui();   // real SDL message boxes for MsgDialog/ErrorDialog (#347)
        fprintf(stderr, "[app] dialog backend installed.\n");
    } else {
        fprintf(stderr, "[app] SDL dialog backend disabled; using headless auto-dismiss.\n");
    }
#endif
}

// Register the live renderer, boot the title at `app0_root`, and run its frame loop on its own
// thread. Returns false with *err set on a link/map/stub failure.
//
// At most one call per process succeeds OR fails: the guard latches on the attempt, not the result,
// so a caller that ignored a failure cannot corrupt g_prog by trying again. Callers reach the
// second-title case through relaunch_with_dump() instead.
static bool start_guest(const std::string& app0_root, std::string* err) {
    if (g_boot_attempted) {
        if (err) *err = g_guest_started ? "a game is already running"
                                        : "this process has already used its one boot attempt";
        return false;
    }
    g_boot_attempted = true;
    // Per-title guest launch arguments (#2973): Unity titles need `-force-gfx-direct` to reach
    // their frame loop in this app (the MT gfx-jobs handshake is not emulated yet — RENDER_LOOP.md);
    // some titles must NOT receive it, so this is explicitly configured, never a silent default.
    // The user's own environment wins over the config file, matching the app_config precedence.
    if (!getenv("PROSPER_GUEST_ARGS")) {
        const prosper::frontend::AppConfig cfg = load_app_config();
        const std::string args = prosper::frontend::guest_args_for(
            cfg, prosper::frontend::title_id_from_app0_path(app0_root));
        if (!args.empty()) {
        set_environment("PROSPER_GUEST_ARGS", args.c_str());
        g_guest_args_app_set = true;
        fprintf(stderr, "[app] guest args (config): %s\n", args.c_str());
        }
    }
#ifdef PROSPER_HAVE_LIVE_RENDERER
    prosper::frontend::register_live_renderer(
        getenv("PROSPER_FRAME_DIR") ? getenv("PROSPER_FRAME_DIR") : ".",
        getenv("PROSPER_APP_DUMP_FRAMES") != nullptr,
        prosper::frontend::title_id_from_app0_path(app0_root));
#else
    fprintf(stderr, "[app] built without the live renderer; the window will stay blank.\n");
#endif
    if (!boot_program(app0_root, g_prog, err, install_host_backends)) return false;
    g_guest_started = true;
    g_guest_thread = std::thread([]{
        const BootResult result = run_entry(g_prog.imgs[0]);
        fprintf(stderr,
                "[app] guest thread ended: kind=%d detail=%s rip=0x%llx addr=0x%llx "
                "rax=0x%llx rbx=0x%llx rdi=0x%llx rsi=0x%llx rdx=0x%llx "
                "rbp=0x%llx rsp=0x%llx\n",
                result.kind, result.detail.c_str(),
                static_cast<unsigned long long>(result.fault_rip),
                static_cast<unsigned long long>(result.fault_addr),
                static_cast<unsigned long long>(result.rax),
                static_cast<unsigned long long>(result.rbx),
                static_cast<unsigned long long>(result.rdi),
                static_cast<unsigned long long>(result.rsi),
                static_cast<unsigned long long>(result.rdx),
                static_cast<unsigned long long>(result.rbp),
                static_cast<unsigned long long>(result.rsp));
        if (result.kind != 0)
            dump_guest_exception_trace();
        for (uint64_t address : result.backtrace)
            fprintf(stderr, "[app] guest backtrace: 0x%llx (%s)\n",
                    static_cast<unsigned long long>(address),
                    describe_code_address(address).c_str());
    });   // runs the guest frame loop
    fprintf(stderr, "[app] guest booted; presenting its frames.\n");
    return true;
}

// The host folder picker. SDL may deliver the result on another thread, so the callback only parks
// the chosen path; the event loop consumes it between frames, where booting is safe.
std::mutex g_picked_mutex;
std::string g_picked_path;
bool g_picker_open = false;

static void picked_folder_cb(void* /*userdata*/, const char* const* filelist, int /*filter*/) {
    std::lock_guard<std::mutex> lock(g_picked_mutex);
    g_picker_open = false;
    if (!filelist) { fprintf(stderr, "[app] folder picker failed: %s\n", SDL_GetError()); return; }
    if (!filelist[0]) { fprintf(stderr, "[app] folder picker cancelled.\n"); return; }
    g_picked_path = filelist[0];
}

// Returns true when this call actually opened a dialog, so a caller can arm per-request state only
// when its request is the one outstanding.
static bool open_folder_picker(SDL_Window* win) {
    {
        std::lock_guard<std::mutex> lock(g_picked_mutex);
        if (g_picker_open) return false;   // one dialog at a time
        g_picker_open = true;
    }
    SDL_ShowOpenFolderDialog(picked_folder_cb, nullptr, win, nullptr, /*allow_many=*/false);
    return true;
}

// Take the path this process was started from. Each platform has its own authoritative answer;
// argv[0] is only the fallback, since it is a valid execv target just when it carries a path and the
// working directory has not moved.
static std::string this_executable(const char* argv0) {
#if defined(_WIN32)
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, sizeof buf);
    if (n > 0 && n < sizeof buf) return std::string(buf, n);
#elif defined(__APPLE__)
    // There is no /proc on macOS. _NSGetExecutablePath may hand back an unresolved path (symlinks,
    // "." components), which execv accepts.
    char buf[4096];
    uint32_t n = sizeof buf;
    if (_NSGetExecutablePath(buf, &n) == 0) return std::string(buf);
#else
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n > 0) { buf[n] = '\0'; return std::string(buf, (size_t)n); }
#endif
    return argv0 ? std::string(argv0) : std::string();
}

// Start a second process for `app0_root`, keeping this run's other arguments. `--dump` is parsed
// last-wins, and a positional path only applies while no dump was given, so appending one always
// selects the new game without having to filter what the user originally passed.
//
// A second in-process boot is not available: the app cannot tear down a running guest (#352). Until
// that lands, a fresh process is the honest way to switch titles.
static bool relaunch_with_dump(int argc, char** argv, const std::string& app0_root) {
    const std::string exe = this_executable(argc > 0 ? argv[0] : nullptr);
    if (exe.empty()) return false;
    std::vector<std::string> args;
    args.push_back(exe);
    for (int i = 1; i < argc; i++) args.emplace_back(argv[i]);
    args.emplace_back("--dump");
    args.push_back(app0_root);
    // If THIS process authored PROSPER_GUEST_ARGS from the config (see start_guest), the value
    // belongs to the title that is shutting down — a relaunch inherits environ verbatim, so leaving
    // it would apply the previous title's args to the new one (whose config entry may differ or
    // forbid it). Removing it lets the child re-resolve from its own title's settings. A
    // USER-authored value is left alone: the documented precedence is that the environment wins
    // over the file.
    if (g_guest_args_app_set) clear_environment("PROSPER_GUEST_ARGS");
#ifdef _WIN32
    // CreateProcess takes a single command line, so every argument is quoted — dump paths routinely
    // contain spaces, and a trailing backslash would otherwise escape the closing quote.
    std::string cmdline = prosper::frontend::windows_command_line(args);
    STARTUPINFOA si{}; si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr,
                        &si, &pi)) {
        fprintf(stderr, "[app] CreateProcess failed: %lu\n", (unsigned long)GetLastError());
        return false;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
#else
    std::vector<char*> cargv;
    cargv.reserve(args.size() + 1);
    for (auto& a : args) cargv.push_back(a.data());
    cargv.push_back(nullptr);
    // posix_spawn rather than fork+exec, for two reasons. It avoids duplicating the page tables of a
    // guest holding multi-gigabyte fixed mappings, which is slow and can hit ENOMEM; and on every
    // libc this builds against (glibc >= 2.24, musl, Darwin) it reports an exec failure — a bad
    // path, a binary replaced underneath us — synchronously, so a failed relaunch can be shown to
    // the user instead of this process quitting into nothing. POSIX permits an implementation to
    // return 0 and let the child exit 127 instead; there the relaunch would degrade to the old
    // silent behaviour rather than misbehave.
    pid_t child = 0;
    const int rc = ::posix_spawn(&child, cargv[0], nullptr, nullptr, cargv.data(), environ);
    if (rc != 0) {
        fprintf(stderr, "[app] could not start %s: %s\n", cargv[0], strerror(rc));
        return false;
    }
    return true;
#endif
}

} // namespace

int main(int argc, char** argv) {
    // Line-buffer stdout: the boot/loader diagnostics go through printf (stdout), and under a
    // file redirect stdout block-buffers — every boot-time line then flushes at exit and lands
    // AFTER hours' worth of unbuffered stderr in a merged `> log 2>&1`, making the log read as
    // if the modules loaded at shutdown (misled a #2981 FMV measurement session).
    setvbuf(stdout, nullptr, _IOLBF, 0);
    bool testPattern = false; int exitAfter = 0; uint32_t winW = 1280, winH = 720;
    prosper::frontend::AppPresentMode requestedPresentMode = prosper::frontend::AppPresentMode::fifo;
    std::string dump;
    // The game library (#1471): where to look, and whether to just print what is there and exit.
    std::string gamesDirFlag;
    std::string setGamesDir;
    bool setGamesDirSeen = false;
    bool listGames = false;
    bool showFps = false;   // --fps
    // Whether to offer the host folder picker at startup (#1469); resolved by should_pick_at_startup.
    prosper::frontend::StartupPickInputs pick{};
    pick.bare_launch = (argc <= 1);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--test-pattern") testPattern = true;
        // --fps: draw the framerate over the running title. OFF by default -- a clean window is the
        // right default, and a HUD that moved would change what a comparison screenshot shows.
        else if (a == "--fps") showFps = true;
        else if (a == "--frames" && i + 1 < argc) exitAfter = atoi(argv[++i]);   // present N frames then exit (CI/smoke)
        // --dump is LAST-WINS, and the positional form below applies only while no dump was given.
        // relaunch_with_dump() depends on that: it appends "--dump <new title>" to this run's own
        // arguments and needs the appended one to override whatever selected the current game.
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];                // boot the game at this app0 dir
        else if (a == "--volume" && i + 1 < argc) {
            // Percent, 0-100. Clamped rather than rejected so a typo cannot deafen.
            g_volume_percent = atoi(argv[++i]);
            if (g_volume_percent < 0) g_volume_percent = 0;
            if (g_volume_percent > 100) g_volume_percent = 100;
        }
        else if (a == "--pick") pick.forced = true;         // open the folder picker at startup
        else if (a == "--no-pick") pick.suppressed = true;  // never open it (scripts, CI, kiosk runs)
        else if (a == "--games-dir") {                       // where the titles are, this run only
            if (i + 1 >= argc || !*argv[i + 1]) {
                fprintf(stderr, "prosper-app: --games-dir requires a path\n");
                return 2;
            }
            gamesDirFlag = argv[++i];
        }
        else if (a == "--list-games") listGames = true;      // print the library and exit (headless)
        else if (a == "--set-games-dir") {                   // persist and exit
            if (i + 1 >= argc) {
                fprintf(stderr, "prosper-app: --set-games-dir requires a path (\"\" clears it)\n");
                return 2;
            }
            setGamesDir = argv[++i];
            setGamesDirSeen = true;   // an empty value is meaningful here: it clears the setting
        }
        else if (a == "--present-mode") {
            if (i + 1 >= argc ||
                !prosper::frontend::parse_present_mode(argv[++i], requestedPresentMode)) {
                fprintf(stderr, "prosper-app: --present-mode requires fifo, mailbox, or immediate\n");
                return 2;
            }
        }
        else if (a == "--record") {
            if (i + 1 >= argc) {
                fprintf(stderr, "prosper-app: --record requires a path\n");
                return 2;
            }
            if (!set_environment("PROSPER_PAD_RECORD", argv[++i])) {
                fprintf(stderr, "prosper-app: failed to set PROSPER_PAD_RECORD\n");
                return 2;
            }
        }
        else if (a == "--record-axis") {
            if (i + 1 >= argc) {
                fprintf(stderr, "prosper-app: --record-axis requires flip or pad-read\n");
                return 2;
            }
            const std::string axis = argv[++i];
            if (axis != "flip" && axis != "pad-read") {
                fprintf(stderr, "prosper-app: --record-axis requires flip or pad-read\n");
                return 2;
            }
            if (!set_environment("PROSPER_PAD_RECORD_AXIS", axis.c_str())) {
                fprintf(stderr, "prosper-app: failed to set PROSPER_PAD_RECORD_AXIS\n");
                return 2;
            }
        }
        else if (a == "--hdr") {
            // Advertise an HDR-capable display to the guest (sceVideoOut capability + output
            // status). Default is SDR — the mode most users' displays and captures expect, and
            // the path where titles apply their own tonemapping. Presentation itself is
            // unchanged (SDR swapchain): a title that commits to PQ output will look wrong —
            // this is an A/B knob, not an HDR presentation path.
            if (!set_environment("PROSPER_HDR", "1")) {
                fprintf(stderr, "prosper-app: failed to set PROSPER_HDR\n");
                return 2;
            }
        }
        else if (a == "--display-mode" && i + 1 < argc) {
            // Which display prosper tells the guest it is attached to (#3017). `legacy` (the
            // default, so this flag is never needed to keep today's behaviour) always advertises
            // 1920x1080 @ 59.94. `host` derives resolution and refresh from the real display.
            // `host-high-refresh` additionally allows the 119.88 Hz enumerant, whose VALUE prosper
            // cannot yet defend from primary evidence and which no title here has been checked
            // against for flip-rate-dependent logic -- see display_mode.hpp.
            const std::string requested = argv[++i];
            prosper::hle::graphics::DisplayModePolicy parsed{};
            if (!prosper::hle::graphics::parse_display_mode_policy(requested, parsed)) {
                fprintf(stderr, "prosper-app: --display-mode must be legacy, host, or "
                                "host-high-refresh (got \"%s\")\n", requested.c_str());
                return 2;
            }
            if (!set_environment("PROSPER_DISPLAY_MODE", requested.c_str())) {
                fprintf(stderr, "prosper-app: failed to set PROSPER_DISPLAY_MODE\n");
                return 2;
            }
        }
        else if (a[0] != '-' && dump.empty()) dump = a;                          // positional dump path
    }
    pick.has_dump = !dump.empty();
    pick.test_pattern = testPattern;

    // --set-games-dir: record the games directory for future launches and exit. Persisting is always
    // an explicit act — nothing here infers a library location from a folder the user happened to open,
    // because guessing wrong would silently point the library somewhere they never chose.
    if (setGamesDirSeen) {
        prosper::frontend::AppConfig cfg = load_app_config();   // keeps keys this build does not know
        cfg.games_dir = prosper::frontend::strip_trailing_separators(setGamesDir);
        // Warn but still store: configuring a path before mounting it is plausible, and refusing
        // would be more annoying than saying so.
        if (!cfg.games_dir.empty() && !host_path_probe().is_dir(cfg.games_dir))
            fprintf(stderr, "prosper-app: warning: %s is not a directory\n", cfg.games_dir.c_str());
        if (!save_app_config(cfg)) return 1;
        if (cfg.games_dir.empty())
            fprintf(stderr, "prosper-app: games directory cleared (%s)\n", app_config_path().c_str());
        else
            fprintf(stderr, "prosper-app: games directory set to %s (%s)\n", cfg.games_dir.c_str(),
                    app_config_path().c_str());
        return 0;
    }

    // The games directory, by the documented precedence: --games-dir, then PROSPER_GAMES_DIR, then the
    // persisted setting. Resolved before anything opens a window so --list-games stays headless.
    const prosper::frontend::AppConfig appConfig = load_app_config();
    const char* gamesDirEnv = getenv("PROSPER_GAMES_DIR");
    std::string gamesDir = prosper::frontend::resolve_games_dir(
        gamesDirFlag, gamesDirEnv ? gamesDirEnv : "", appConfig);

    // --list-games: print the library as plain text and exit, with no window, no Vulkan and no guest.
    // One tab-separated record per line on stdout (content id, display name, app0 path) so a script or
    // an agent can consume it; everything explanatory goes to stderr.
    if (listGames) {
        if (gamesDir.empty()) {
            fprintf(stderr, "prosper-app: no games directory. Pass --games-dir <path>, set "
                            "PROSPER_GAMES_DIR, or record one in %s\n",
                    app_config_path().empty() ? "the settings file" : app_config_path().c_str());
            return 2;
        }
        if (!host_path_probe().is_dir(gamesDir)) {
            // Distinguish a wrong path from a real but empty library: both would otherwise print
            // "0 title(s)" and exit 1, which hides a typo.
            fprintf(stderr, "prosper-app: not a directory: %s\n", gamesDir.c_str());
            return 2;
        }
        const std::vector<prosper::frontend::GameEntry> games =
            prosper::frontend::scan_game_library(gamesDir, host_path_probe(), host_library_io());
        for (const auto& g : games)
            printf("%s\t%s\t%s\n", g.title_id.c_str(), g.title_name.c_str(), g.app0_root.c_str());
        fprintf(stderr, "prosper-app: %zu title(s) in %s\n", games.size(), gamesDir.c_str());
        return games.empty() ? 1 : 0;
    }

    // #3017: publish the host's REAL display mode before anything boots, so the VideoOut layer can
    // derive what it advertises to the guest instead of hardcoding 1920x1080 @ 59.94 Hz. Publishing
    // is unconditional and inert on its own -- PROSPER_DISPLAY_MODE (default `legacy`) decides
    // whether the derivation actually uses it, so a title that works today is untouched until
    // somebody opts in.
    //
    // Position is load-bearing: start_guest() below spawns the guest thread, and the VideoOut layer
    // resolves its mode ONCE on first use. Publishing after the boot would make which mode the guest
    // sees depend on whether it asked before we wrote the variable -- a race, and one that would
    // usually resolve the wrong way. SDL_InitSubSystem rather than the authoritative SDL_Init
    // further down: video is not up yet here, and a failure must cost only the derivation (the
    // documented fallback is exactly the hardcoded mode prosper always used), never the boot.
    if (SDL_InitSubSystem(SDL_INIT_VIDEO)) {
        const SDL_DisplayMode* host_mode = SDL_GetDesktopDisplayMode(SDL_GetPrimaryDisplay());
        if (host_mode && host_mode->w > 0 && host_mode->h > 0 && host_mode->refresh_rate > 0.0f) {
            char published[64];
            snprintf(published, sizeof published, "%dx%d@%.4f",
                     host_mode->w, host_mode->h, (double)host_mode->refresh_rate);
            if (set_environment("PROSPER_HOST_DISPLAY_MODE", published))
                fprintf(stderr, "[app] host display: %s\n", published);
            else
                fprintf(stderr, "[app] failed to publish PROSPER_HOST_DISPLAY_MODE\n");
        } else {
            fprintf(stderr, "[app] host display mode unavailable (%s); VideoOut keeps its "
                            "1920x1080 @ 59.94 default.\n", SDL_GetError());
        }
    }

    // Boot the game (unless test-pattern): the shared start_guest() path registers the live renderer
    // so the guest's GPU submits composite to frames on the present layer, boots through
    // boot_program, and runs the guest on its own thread while this thread owns the window +
    // present. Reaching the frame loop needs PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct
    // in the environment (same as boot_trace).
    //
    // This is the argv path and it keeps its original position: the guest is up before the window
    // exists. A title opened later (drop / picker, #1469) boots from inside the event loop instead,
    // through this same start_guest().
    if (!testPattern && !dump.empty()) {
        std::string err;
        if (!start_guest(dump, &err)) { fprintf(stderr, "[app] boot failed: %s\n", err.c_str()); return 1; }
    } else if (!testPattern) {
        // The library view that will draw these is #1471 stage 2; for now report what was found so a
        // misconfigured games_dir is visible without waiting for the UI.
        if (!gamesDir.empty()) {
            const size_t found = prosper::frontend::scan_game_library(
                gamesDir, host_path_probe(), host_library_io()).size();
            fprintf(stderr, "[app] games directory %s holds %zu title(s); --list-games prints them.\n",
                    gamesDir.c_str(), found);
        }
        fprintf(stderr, "[app] no game given; the window opens empty and can be given one "
                        "(drop a game folder on it, or press Ctrl+O).\n");
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) { fprintf(stderr, "[app] SDL_Init: %s\n", SDL_GetError()); return 1; }
#ifdef __APPLE__
    // There is no system Vulkan loader on macOS; point SDL at MoltenVK so SDL_Vulkan_* uses the same
    // driver this binary links. PROSPER_VULKAN_LIB overrides the path; default resolves via the
    // executable's rpath (CMake links MoltenVK with an rpath, so the dylib sits beside/near the app).
    if (!SDL_Vulkan_LoadLibrary(getenv("PROSPER_VULKAN_LIB") ? getenv("PROSPER_VULKAN_LIB") : "libMoltenVK.dylib")) {
        fprintf(stderr, "[app] SDL_Vulkan_LoadLibrary(MoltenVK): %s\n", SDL_GetError());
        fprintf(stderr, "[app] set PROSPER_VULKAN_LIB=/path/to/libMoltenVK.dylib\n");
        return 1;
    }
#endif
    // Title: "prosper - <game name>" for a booted game, else a label saying what the empty window
    // is waiting for. A title opened later replaces this (#1469).
    std::string title = window_title_for(dump, testPattern);
    fprintf(stderr, "[app] window title: \"%s\"\n", title.c_str());
    SDL_Window* win = SDL_CreateWindow(title.c_str(), (int)winW, (int)winH, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!win) { fprintf(stderr, "[app] SDL_CreateWindow: %s\n", SDL_GetError()); return 1; }

    Vk vk;
    // Present unification (#1270): prefer the renderer's shared device for real game boots so we can
    // GPU-blit its front-buffer image. The test pattern, PROSPER_APP_GPU_PRESENT=0, and any adoption
    // failure fall back to a private device + CPU pixels.
    const bool wantGpuPresent = prosper::frontend::request_gpu_present(
        getenv("PROSPER_APP_GPU_PRESENT"), testPattern, !dump.empty());
    if (!wantGpuPresent || !try_adopt_shared_present(vk, win)) {
        if (!create_instance(vk, win) || !pick_device(vk)) return 1;
    }
    // Initial swapchain sized to the window; recreated on resize / out-of-date.
    { int dw = 0, dh = 0; SDL_GetWindowSizeInPixels(win, &dw, &dh);
      if (!create_swapchain(vk, (uint32_t)dw, (uint32_t)dh, requestedPresentMode)) return 1; }

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex = vk.qfamily;
    vkCreateCommandPool(vk.device, &cpi, nullptr, &vk.cmdPool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = vk.cmdPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(vk.device, &cai, &vk.cmd);
    if (!replace_present_sync(vk)) {
        fprintf(stderr, "[app] could not create present synchronization\n");
        return 1;
    }

    fprintf(stderr, "[app] window up (%s). Close the window or press Esc to quit.\n",
            testPattern ? "test-pattern" : "waiting for guest frames");
#ifdef _WIN32
    HWND hwnd = static_cast<HWND>(SDL_GetPointerProperty(
        SDL_GetWindowProperties(win), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
    if (hwnd) {
        ShowWindow(hwnd, SW_SHOWNORMAL);
        UpdateWindow(hwnd);
        SetForegroundWindow(hwnd);
    }
#endif

    // Keyboard controls augment SDL pad 0; the fallback keeps physical pads and their analog state.
    prosper::input::pad_set_backend(&g_keyboard_pad);
    fprintf(stderr, "[app] keyboard: WASD/Arrows=move  J/Space=Cross(jump)  K=Square(attack)  L=Circle  "
                    "I=Triangle  U/O=L1/R1  Y/H=L2/R2  Enter=Options  "
                    "Pause/F10=pause  F11/Alt+Enter=fullscreen  Esc=quit\n");
    fprintf(stderr, "[app] F9 = grab the current frame: writes a replayable .prgbundle + a .bmp "
                    "screenshot named frame_grab_<titleId>_<date>-<time>-<ms>.* (to "
                    "PROSPER_CAPTURE_DIR, default cwd) for gpu_replay debugging "
                    "(brief hitch on press; PROSPER_CAPTURE_FRAMES>1 grabs an animation).\n");
    fprintf(stderr, "[app] F8 = capture 5 seconds before + 5 seconds after the press into a bounded "
                    ".prperf performance report (to PROSPER_CAPTURE_DIR, default cwd).\n");

    const bool frameTrace = getenv("PROSPER_APP_FRAME_TRACE") != nullptr;
    const char* stallDumpEnv = getenv("PROSPER_APP_STALL_DUMP_MS");
    const int stallDumpMs = stallDumpEnv ? std::max(0, atoi(stallDumpEnv)) : 0;
    const char* timedDumpEnv = getenv("PROSPER_APP_GUEST_DUMP_MS");
    const int timedDumpMs = timedDumpEnv ? std::max(0, atoi(timedDumpEnv)) : 0;
    const char* timedDumpIntervalEnv = getenv("PROSPER_APP_GUEST_DUMP_INTERVAL_MS");
    const int timedDumpIntervalMs = timedDumpIntervalEnv ?
        std::max(0, atoi(timedDumpIntervalEnv)) : 0;
    const char* timedDumpPath = getenv("PROSPER_APP_GUEST_DUMP_PATH");
    const char* timedDumpPthreadEnv = getenv("PROSPER_APP_GUEST_DUMP_PTHREAD");
    const uint64_t timedDumpPthread = timedDumpPthreadEnv ?
        strtoull(timedDumpPthreadEnv, nullptr, 0) : 0;
    const auto loopStarted = std::chrono::steady_clock::now();
    auto lastFrameProgress = loopStarted;
    auto nextTimedDump = loopStarted + std::chrono::milliseconds(timedDumpMs);
    prosper::frontend::PresentedFrameCounter shown(exitAfter);
    uint64_t lastFrameSeq = ~0ull, patFrame = 0;
    bool havePresentedGuestFlip = false;
    uint64_t lastPresentedGuestFlip = 0;
    int gpuPrevSlot = -1;   // #1270: the GPU scanout slot presented last frame (released after its read)
    unsigned timedDumpCount = 0;
    // Interactive frame grab (F9): output dir plus the namer that claims one stamped, title-tagged
    // name pair per press (frame_grab_naming.hpp). The old per-process counter (frame_grab_001) let a
    // second title played in the same directory overwrite the first title's captures, and let an
    // aborted grab's .bmp sit beside a same-named .prgbundle from an earlier boot.
    const std::string grabDir = getenv("PROSPER_CAPTURE_DIR") ? getenv("PROSPER_CAPTURE_DIR") : ".";
    prosper::frontend::FrameGrabNamer grabNamer;
    grabNamer.set_directory(grabDir);
    CaptureTitle activeCaptureTitle = capture_title_for(dump);
    {
        grabNamer.set_title(activeCaptureTitle.id, activeCaptureTitle.label);
    }
    // ---- Human-authored render snapshots (F6 "correct" / F7 "wrong") -------------------------
    // The person playing is the oracle. Each press writes the presented frame plus a manifest
    // record anchored on the PAD FLIP ORDINAL, the same axis PROSPER_PAD_RECORD writes routes
    // against -- so a recorded route and the snaps taken during it index each other, and an
    // automated run can replay to the exact moment a human passed judgement.
    //
    // Read-only accessor: it must not be the call that establishes the pad flip origin. See the
    // comment on prosper_pad_flip_ordinal() in hle_pad.cpp for what that would corrupt.
    const std::string snapDir = getenv("PROSPER_SNAP_DIR") ? getenv("PROSPER_SNAP_DIR") : grabDir;
    std::optional<prosper::frontend::SnapVerdict> pendingSnapVerdict;
    prosper::frontend::SnapMode pendingSnapMode = prosper::frontend::SnapMode::anchor;
    uint32_t snapCount = 0;
    // Check side: capture the presented frame as each authored anchor arrives. Sorted ascending by
    // the parser, so one forward-only cursor covers the run -- and a target already passed when the
    // list is loaded is simply never armed, rather than firing late against the wrong scene.
    const std::vector<int64_t> snapFlipTargets =
        prosper::frontend::parse_snap_flip_list(getenv("PROSPER_SNAP_AT_FLIPS"));
    size_t nextSnapTarget = 0;
    std::optional<int64_t> pendingActualTarget;
    if (!snapFlipTargets.empty())
        std::fprintf(stderr, "[snap] replay capture armed for %zu anchor(s), first at pad flip %lld\n",
                     snapFlipTargets.size(), static_cast<long long>(snapFlipTargets.front()));

    prosper::perf::InteractivePerformanceCapture& perfCapture =
        prosper::perf::interactive_performance_capture();
    const char* automaticPerfEnv = getenv("PROSPER_PERF_CAPTURE_AFTER_MS");
    prosper::frontend::ElapsedPerformanceCaptureTrigger automaticPerfCapture(
        prosper::frontend::parse_performance_capture_delay_ns(automaticPerfEnv));
    if (automaticPerfEnv) {
        if (automaticPerfCapture.delay_ns()) {
            std::fprintf(stderr,
                "[perf] automatic capture scheduled by PROSPER_PERF_CAPTURE_AFTER_MS=%s "
                "(one attempt after %llu ms of app-loop time)\n",
                automaticPerfEnv,
                static_cast<unsigned long long>(automaticPerfCapture.delay_ns() / 1'000'000));
        } else {
            std::fprintf(stderr,
                "[perf] ignoring malformed PROSPER_PERF_CAPTURE_AFTER_MS=\"%s\" "
                "(expected positive decimal milliseconds without overflow)\n",
                automaticPerfEnv);
        }
    }
    bool perfCaptureWasAutomatic = false;
    auto arm_performance_capture = [&](bool automatic, uint64_t armedAt,
                                       uint64_t automaticElapsedMs = 0) {
        if (perfCapture.sample_due(armedAt)) {
            perfCapture.observe_sample(prosper::perf::collect_process_sample(
                armedAt, gpu::present_count(),
                prosper::frontend::rendered_frame_counter(
                    vk.gpu_present, gpu::present_frame_seq()),
                shown));
        }
        const prosper::perf::CaptureArmResult armed = perfCapture.arm(
            grabDir, activeCaptureTitle.id, activeCaptureTitle.label,
            prosper::perf::build_revision(), armedAt,
            std::chrono::system_clock::now());
        if (armed.ok) perfCaptureWasAutomatic = automatic;
        if (automatic) {
            if (armed.ok) {
                std::fprintf(stderr,
                    "[perf] automatic trigger #%u fired at %llu ms for %s: retained %zu "
                    "pre-trigger samples; collecting %.1f seconds after the trigger\n",
                    armed.index, static_cast<unsigned long long>(automaticElapsedMs),
                    activeCaptureTitle.label.empty() ? "the current process"
                                                     : activeCaptureTitle.label.c_str(),
                    armed.pre_samples, armed.post_seconds);
            } else {
                std::fprintf(stderr,
                    "[perf] automatic trigger #%u fired at %llu ms but was not armed: %s\n",
                    armed.index, static_cast<unsigned long long>(automaticElapsedMs),
                    armed.error.c_str());
            }
        } else if (armed.ok) {
            std::fprintf(stderr,
                "[perf] F8 #%u armed for %s: retained %zu pre-trigger samples; "
                "collecting %.1f seconds after the press\n",
                armed.index,
                activeCaptureTitle.label.empty() ? "the current process"
                                                 : activeCaptureTitle.label.c_str(),
                armed.pre_samples, armed.post_seconds);
        } else {
            std::fprintf(stderr, "[perf] F8 #%u not armed: %s\n",
                         armed.index, armed.error.c_str());
        }
    };
    // 0 = keep the built-in default; the timeline clamps whatever is supplied to 64..3072 MiB.
    // Parse strictly, matching the two checked parses of this same variable in gpu_timeline.cpp. An
    // unchecked strtoul truncates into uint32_t, so 4294967360 would arrive as 64 — silently the
    // MINIMUM budget when the user asked for the largest, which is the opposite of the intent and
    // would look like the fix not working.
    uint32_t grabBundleMaxMb = 0u;
    if (const char* mb = getenv("PROSPER_CAPTURE_BUNDLE_MAX_MB")) {
        char* end = nullptr;
        errno = 0;
        const unsigned long parsed = strtoul(mb, &end, 0);
        if (!errno && end != mb && end && !*end && parsed <= UINT32_MAX)
            grabBundleMaxMb = static_cast<uint32_t>(parsed);
        else
            fprintf(stderr, "[grab] ignoring malformed PROSPER_CAPTURE_BUNDLE_MAX_MB=\"%s\" "
                            "(expected 64..3072)\n", mb);
    }
    std::string grabNotice;   // transient window-title notice for the last completed F9 grab (#1587)
    std::chrono::steady_clock::time_point grabNoticeUntil{};
    std::string pendingGrabScreenshot;   // non-empty => write the next presented CPU frame to this path
    uint64_t pendingGrabGuestPresent = 0;
    unsigned pendingGrabSuffix = 0;      // the collision suffix this capture owns, 0 when it needed none
    bool pendingGrabReserved = false;    // the pending screenshot path came from a reservation (F9),
                                         // not from PROSPER_CAPTURE_SCREENSHOT* (a configured path)
    // The bundle names THIS frontend reserved, with the collision suffix each one owns. It is an
    // ownership record first and a suffix lookup second: the same outcome channel also reports the
    // env-driven captures (PROSPER_CAPTURE_BUNDLE / …_AT_PRESENT / …_AFTER_GUEST_LOG), whose paths
    // this frontend never created and must not delete or describe as reserved.
    // Bounded: a capture that never reports (the process died) must not accumulate here.
    std::vector<std::pair<std::string, unsigned>> grabReservedBundles;
    // Returns whether THIS frontend reserved `path`, and if so its suffix. "Absent" and "found with
    // suffix 0" are different answers and must not collapse into one.
    auto take_bundle_reservation = [&](const std::string& path, unsigned& suffix) -> bool {
        for (size_t i = 0; i < grabReservedBundles.size(); ++i) {
            if (grabReservedBundles[i].first != path) continue;
            suffix = grabReservedBundles[i].second;
            grabReservedBundles.erase(grabReservedBundles.begin() + static_cast<ptrdiff_t>(i));
            return true;
        }
        suffix = 0;
        return false;
    };
    // The F9 frame grab, callable from the hotkey AND from the headless triggers below (#2233).
    // Extracted verbatim so an unattended capture takes exactly the interactive path -- the
    // supersede accounting and the reservation bookkeeping included. A second implementation for
    // automation would mean a second set of rules for who owns a reserved file, and that is the
    // one part of this that has already been hard to get right.
    auto arm_frame_grab = [&](bool automatic, const char* why) {
        // Every line this path emits is read by an agent, not an operator. `source` names what armed
        // the capture, so a SCHEDULED grab reports its trigger rather than a keypress nobody made --
        // a log line naming F9 on a run with no window sends its reader looking for an operator.
        const char* const source = automatic ? why : "F9";
        std::fprintf(stderr, "[grab] %s bundle grab requested (%s)\n",
                     automatic ? "automatic" : "F9", why);
                // Claim both output names now, from ONE timestamp, with an exclusive create. Doing
                // it here rather than at each write is what binds the two artifacts of this
                // capture together: they are written seconds apart on two different threads, and a
                // name derived independently at each write would let a bundle take a collision
                // suffix its screenshot did not — which is exactly the mismatched pair this
                // change exists to make impossible.
                const prosper::frontend::FrameGrabPaths grab = grabNamer.reserve();
                if (!grab.ok) {
                    std::fprintf(stderr, "[grab] %s #%u: could not reserve capture files: %s\n",
                                 source, grab.index, grab.error.c_str());
                    return;
                }
                // This line names NO file, deliberately. A log line must assert only what is true
                // when it is emitted: at arm time the capture may still abort, the write may fail,
                // and a collision suffix makes the eventual path unequal to any guess. An intention
                // written in the grammar of a result is one a reader — human or agent — cannot tell
                // apart from a result, and this fleet's agents read these logs to find artifacts.
                // The real paths are logged by the writes, once the files exist. Do not "helpfully"
                // restore a filename here.
                std::fprintf(stderr, "%s\n",
                             prosper::frontend::frame_grab_arm_line(
                                 grab.index, grabNamer.title_label(), source).c_str());
                // Honour PROSPER_CAPTURE_BUNDLE_MAX_MB here too (#1587). It was consulted only on
                // the headless/scheduled paths, so the hotkey was pinned to the 2 GiB default with
                // no way to raise it without editing code — and one 3840x2160 frame of a deferred
                // renderer exceeds that, which made F9 unusable on 4K UE titles.
                // Supersede and EXPLAIN, rather than refuse. A press that lands before the
                // previous arm was promoted replaces it, and a replaced arm never runs and never
                // reports — so this frontend is the only thing that can account for the names it
                // reserved. Refusing the press instead was considered and rejected: the flag that
                // says "a grab is in flight" is cleared only by a guest PRESENT, so a hung title
                // would leave F9 dead for the rest of the session — and a hung title is exactly
                // when someone reaches for F9.
                const std::string replaced =
                    prosper::gpu::request_interactive_capture_bundle(grab.bundle, grabBundleMaxMb);
                if (!grab.warning.empty())
                    std::fprintf(stderr, "[grab] %s\n", grab.warning.c_str());
                if (!replaced.empty()) {
                    unsigned dropped = 0;
                    const bool ours = take_bundle_reservation(replaced, dropped);
                    std::error_code ec;
                    // Only our own, and only while still empty: a file with bytes in it belongs
                    // to whoever wrote them.
                    const bool removed = ours && std::filesystem::exists(replaced, ec) && !ec &&
                                         std::filesystem::file_size(replaced, ec) == 0 && !ec &&
                                         std::filesystem::remove(replaced, ec) && !ec;
                    std::fprintf(stderr,
                                 "[grab] this press replaced an armed capture that had not started; "
                                 "it will never report. %s: %s\n",
                                 removed ? "its empty reserved bundle has been removed"
                                         : (ours ? "its reserved bundle is still on disk"
                                                 : "its output path was configured, not reserved, "
                                                   "and was left alone"),
                                 replaced.c_str());
                }
                // A pending screenshot is dropped by this press whichever kind it is, but the two
                // kinds cannot share a sentence: one is a name this frontend reserved and can
                // account for, the other a configured path it never created.
                if (!pendingGrabScreenshot.empty()) {
                    if (pendingGrabReserved) {
                        std::error_code ec;
                        const bool removed =
                            std::filesystem::exists(pendingGrabScreenshot, ec) && !ec &&
                            std::filesystem::file_size(pendingGrabScreenshot, ec) == 0 && !ec &&
                            std::filesystem::remove(pendingGrabScreenshot, ec) && !ec;
                        std::fprintf(stderr,
                                     "[grab] the previous capture's screenshot was still pending and "
                                     "is dropped by this press; %s: %s\n",
                                     removed ? "its empty reserved file has been removed"
                                             : "its reserved file is still on disk",
                                     pendingGrabScreenshot.c_str());
                    } else {
                        std::fprintf(stderr,
                                     "[grab] the pending scheduled screenshot is dropped by this "
                                     "press; nothing was written to %s\n",
                                     pendingGrabScreenshot.c_str());
                    }
                }
                pendingGrabScreenshot = grab.screenshot;
                pendingGrabGuestPresent = gpu::present_count();
                pendingGrabSuffix = grab.suffix;
                pendingGrabReserved = true;
                if (grabReservedBundles.size() >= 32) grabReservedBundles.erase(grabReservedBundles.begin());
                grabReservedBundles.emplace_back(grab.bundle, grab.suffix);
                return;
    };

    // Write one authored snap: the presented pixels, plus an appended JSON Lines manifest record.
    // Appending per snap rather than rewriting an array is deliberate -- an authoring session ends
    // when the person closes the window or kills the process, and a truncated array would lose the
    // whole session's judgements rather than the last one.
    auto flushPendingSnap = [&](const uint8_t* rgba, uint32_t w, uint32_t h) {
        if (!pendingSnapVerdict) return;
        const prosper::frontend::SnapVerdict verdict = *pendingSnapVerdict;
        const prosper::frontend::SnapMode mode = pendingSnapMode;
        pendingSnapVerdict.reset();
        pendingSnapMode = prosper::frontend::SnapMode::anchor;
        const int64_t flip = prosper_pad_flip_ordinal();
        const uint32_t index = snapCount++;
        const std::string name = prosper::frontend::snap_file_name(index, verdict, flip);
        std::error_code ec;
        std::filesystem::create_directories(snapDir, ec);
        const std::string path = (std::filesystem::path(snapDir) / name).string();
        if (!write_frame_bmp(path, rgba, w, h)) {
            std::fprintf(stderr, "[snap] FAILED to write %s -- this judgement was lost\n",
                         path.c_str());
            return;
        }
        const std::string manifest =
            (std::filesystem::path(snapDir) / "snaps.jsonl").string();
        if (FILE* f = fopen(manifest.c_str(), "a")) {
            std::fprintf(f, "%s\n", prosper::frontend::snap_record_line(
                index, verdict, mode, flip, gpu::present_count(), w, h, name,
                grabNamer.title_id()).c_str());
            fclose(f);
        } else {
            std::fprintf(stderr, "[snap] wrote %s but could NOT append to %s -- the image exists "
                                 "with no record, and import will not see it\n",
                         name.c_str(), manifest.c_str());
        }
        std::fprintf(stderr, "%s\n",
                     prosper::frontend::snap_author_line(index, verdict, mode, flip, name).c_str());
    };

    auto flushPendingActual = [&](const uint8_t* rgba, uint32_t w, uint32_t h) {
        if (!pendingActualTarget) return;
        const int64_t target = *pendingActualTarget;
        pendingActualTarget.reset();
        const int64_t actual = prosper_pad_flip_ordinal();
        const std::string name = prosper::frontend::snap_actual_file_name(target, actual);
        std::error_code ec;
        std::filesystem::create_directories(snapDir, ec);
        const std::string path = (std::filesystem::path(snapDir) / name).string();
        if (!write_frame_bmp(path, rgba, w, h)) {
            std::fprintf(stderr, "[snap] FAILED to write %s\n", path.c_str());
            return;
        }
        const std::string manifest =
            (std::filesystem::path(snapDir) / "actuals.jsonl").string();
        if (FILE* f = fopen(manifest.c_str(), "a")) {
            std::fprintf(f, "%s\n", prosper::frontend::snap_actual_record_line(
                target, actual, w, h, name).c_str());
            fclose(f);
        }
        std::fprintf(stderr, "[snap] captured anchor %lld at pad flip %lld -> %s\n",
                     static_cast<long long>(target), static_cast<long long>(actual), name.c_str());
    };

    auto flushGrabScreenshot = [&](const uint8_t* rgba, uint32_t w, uint32_t h) {
        if (pendingGrabScreenshot.empty()) return;
        const bool ok = write_frame_bmp(pendingGrabScreenshot, rgba, w, h);
        char detail[128];
        std::snprintf(detail, sizeof detail, "armed at guest present %llu, written at guest present %llu",
                      static_cast<unsigned long long>(pendingGrabGuestPresent),
                      static_cast<unsigned long long>(gpu::present_count()));
        // Emitted AFTER the write, and it names the file that now exists. The arming line deliberately
        // named none: see the F9 handler.
        if (ok) {
            std::fprintf(stderr, "%s\n",
                         prosper::frontend::frame_grab_write_line(
                             "screenshot", pendingGrabScreenshot, pendingGrabSuffix, detail).c_str());
        } else {
            // Only a reserved path is known to exist and be empty; a configured one may not exist at
            // all, so the two cases cannot share a sentence.
            std::fprintf(stderr, "[grab] screenshot write FAILED (%s); %s: %s\n", detail,
                         pendingGrabReserved ? "the reserved file is still empty"
                                             : "nothing was written to the configured path",
                         pendingGrabScreenshot.c_str());
        }
        pendingGrabScreenshot.clear();
        pendingGrabGuestPresent = 0;
        pendingGrabSuffix = 0;
        pendingGrabReserved = false;
    };
    const uint64_t scheduledScreenshotFrame = prosper::frontend::parse_capture_frame(
        getenv("PROSPER_CAPTURE_SCREENSHOT_AT_FRAME"));
    std::string scheduledScreenshotPath;
    if (scheduledScreenshotFrame) {
        if (const char* configured = getenv("PROSPER_CAPTURE_SCREENSHOT")) {
            scheduledScreenshotPath = configured;
        } else {
            scheduledScreenshotPath = grabDir + "/scheduled_frame_" +
                std::to_string(scheduledScreenshotFrame) + ".bmp";
        }
        // A target, not a result: no arrow (see frame_grab_naming.hpp's logging contract).
        fprintf(stderr, "[grab] screenshot scheduled at host frame %llu; target path %s\n",
                static_cast<unsigned long long>(scheduledScreenshotFrame),
                scheduledScreenshotPath.c_str());
    }

    // --- Headless capture triggers (#2233) -------------------------------------------------
    // F8 and F9 are the richest diagnostics this project has, and the bundle grab could previously
    // be reached ONLY by a human pressing a key. Every agent here runs headless, so the best
    // instrument in the toolbox was the one nobody could aim -- and lanes built weaker ones instead.
    // Both captures are now schedulable on either axis, because they answer different questions:
    // "after N ms" aims at a wall-clock event (a movie starts, a frame rate collapses); "at frame N"
    // aims at a reproducible ordinal already located with a cheap screenshot sweep.
    //
    // All are one-shot and opt-in, and a malformed value disables its own trigger rather than firing
    // at an unintended moment -- see parse_* in capture_schedule.hpp / performance_capture_schedule.hpp.
    prosper::frontend::ElapsedPerformanceCaptureTrigger automaticGrabAfter(
        prosper::frontend::parse_performance_capture_delay_ns(getenv("PROSPER_GRAB_BUNDLE_AFTER_MS")));
    const uint64_t scheduledGrabFrame =
        prosper::frontend::parse_capture_frame(getenv("PROSPER_GRAB_BUNDLE_AT_FRAME"));
    bool scheduledGrabArmed = false;
    const uint64_t scheduledPerfFrame =
        prosper::frontend::parse_capture_frame(getenv("PROSPER_PERF_CAPTURE_AT_FRAME"));
    bool scheduledPerfArmed = false;
    if (automaticGrabAfter.delay_ns())
        std::fprintf(stderr, "[grab] automatic bundle scheduled by PROSPER_GRAB_BUNDLE_AFTER_MS "
                             "(one attempt after %llu ms of app-loop time)\n",
                     (unsigned long long)(automaticGrabAfter.delay_ns() / 1000000));
    if (scheduledGrabFrame)
        std::fprintf(stderr, "[grab] automatic bundle scheduled at host frame %llu\n",
                     (unsigned long long)scheduledGrabFrame);
    if (scheduledPerfFrame)
        std::fprintf(stderr, "[perf] automatic capture scheduled at host frame %llu\n",
                     (unsigned long long)scheduledPerfFrame);
    // RenderDoc, aimed the same two ways as F8/F9 (#3321). Off unless one of these is set, and a
    // malformed value disables its own trigger rather than firing at an unintended moment.
    prosper::frontend::ElapsedPerformanceCaptureTrigger automaticRdocAfter(
        prosper::frontend::parse_performance_capture_delay_ns(getenv("PROSPER_RENDERDOC_AFTER_MS")));
    const uint64_t scheduledRdocFrame =
        prosper::frontend::parse_capture_frame(getenv("PROSPER_RENDERDOC_AT_FRAME"));
    bool scheduledRdocArmed = false;
    bool renderdocCaptureOpen = false;
    // A prosper "frame" is the GUEST's, not the app loop's, and conflating the two produces a
    // capture with no draws in it -- measured, not assumed (#3321). prosper submits the guest's GPU
    // work off the app-loop thread, and the loop itself can spin at 250+ fps while the guest renders
    // far slower, so a span of one loop iteration routinely contains only the presentation blit and
    // vkQueuePresentKHR. The span therefore closes on a guest present, with an iteration cap so a
    // title that never presents again cannot hold a capture open until the process runs out of
    // memory.
    void* renderdocDevicePointer = nullptr;
    uint64_t renderdocOpenedAtGuestPresent = 0;
    uint64_t renderdocOpenIterations = 0;
    // `?:` was a GNU extension and the repo's only use; spelled out so MSVC and -Wpedantic are clean.
    // Note the contract differs from the two TRIGGERS deliberately: a malformed trigger disables
    // itself, because firing at an unintended moment is worse than not firing. This is a safety CAP,
    // not a trigger -- disabling it would mean an unbounded span -- so a malformed or zero value
    // falls back to the default, and says so rather than doing it quietly.
    // How many GUEST presents the span covers, default 1 (#3321). One is right for a title whose
    // frame is self-contained, and demonstrably wrong for one that builds its screen across several
    // frames: Stray's title screen renders its 3D element in a frame that carries no UI, so a
    // one-present capture holds a cat and no menu while the screen shows a menu and no cat. Widening
    // the span is the only way to see a whole cycle. Same fallback-and-announce contract as the cap
    // below -- this bounds a span, it does not aim one.
    const uint64_t renderdocPresentsParsed =
        prosper::frontend::parse_capture_frame(getenv("PROSPER_RENDERDOC_PRESENTS"));
    const uint64_t renderdocPresents = renderdocPresentsParsed ? renderdocPresentsParsed : 1;
    if (const char* np = getenv("PROSPER_RENDERDOC_PRESENTS"); np && !renderdocPresentsParsed)
        std::fprintf(stderr, "[renderdoc] ignoring malformed PROSPER_RENDERDOC_PRESENTS=\"%s\" "
                             "(expected a positive present count); spanning %llu\n",
                     np, (unsigned long long)renderdocPresents);
    const uint64_t renderdocMaxItersParsed =
        prosper::frontend::parse_capture_frame(getenv("PROSPER_RENDERDOC_MAX_ITERS"));
    const uint64_t renderdocMaxIterations = renderdocMaxItersParsed ? renderdocMaxItersParsed : 2000;
    if (const char* mi = getenv("PROSPER_RENDERDOC_MAX_ITERS"); mi && !renderdocMaxItersParsed)
        // Echo the offending value and the accepted shape, matching PROSPER_CAPTURE_BUNDLE_MAX_MB
        // at main.cpp:1750 -- the repo's existing convention for a malformed BOUND, as opposed to a
        // malformed trigger. "not usable" without the value leaves the author guessing which of
        // their variables was wrong.
        std::fprintf(stderr, "[renderdoc] ignoring malformed PROSPER_RENDERDOC_MAX_ITERS=\"%s\" "
                             "(expected a positive iteration count); using the default cap of "
                             "%llu\n",
                     mi, (unsigned long long)renderdocMaxIterations);
    const bool renderdocWanted = scheduledRdocFrame != 0 || automaticRdocAfter.delay_ns() != 0;
    if (renderdocWanted) {
        auto& rdoc = prosper::frontend::RenderDocCapture::instance();
        if (!rdoc.available()) {
            // Say this at ARM time, not at fire time. A route that runs for four minutes before its
            // trigger only to report that RenderDoc was never loaded has wasted the whole run; the
            // configuration error is knowable at startup, so it is reported at startup.
            std::fprintf(stderr, "[renderdoc] REQUESTED BUT UNAVAILABLE: %s\n",
                         rdoc.unavailable_reason().c_str());
        } else {
            // Never leave this at RenderDoc's default, which is under /tmp -- see the header.
            rdoc.set_path_template(grabDir + "/prosper_frame");
            if (renderdocPresents != 1)
                std::fprintf(stderr, "[renderdoc] span set to %llu guest presents\n",
                             (unsigned long long)renderdocPresents);
            if (scheduledRdocFrame)
                std::fprintf(stderr, "[renderdoc] capture scheduled at host frame %llu (into %s)\n",
                             (unsigned long long)scheduledRdocFrame, grabDir.c_str());
            if (automaticRdocAfter.delay_ns())
                std::fprintf(stderr, "[renderdoc] capture scheduled %llu ms into the app loop "
                                     "(into %s)\n",
                             (unsigned long long)(automaticRdocAfter.delay_ns() / 1000000),
                             grabDir.c_str());
        }
    }
    // Opening a capture is one call, but it is worth one place: both triggers report which axis
    // fired, because "at frame 900" and "after 30000 ms" land at different moments on a route whose
    // frame rate is the thing under investigation.
    // Resolve the RENDERER's instance, not prosper-app's -- see renderdoc_capture.hpp. Resolved at
    // fire time rather than at startup because the renderer publishes its context when the guest
    // boots, which is long after these triggers are parsed.
    auto renderdoc_device_pointer = []() -> void* {
        const prosper::gpu::SharedVulkanContext shared = prosper::gpu::shared_vulkan_context();
        if (!shared.valid() || !shared.instance) return nullptr;
        return RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(shared.instance);
    };
    auto arm_renderdoc_capture = [&](const char* why) {
        auto& rdoc = prosper::frontend::RenderDocCapture::instance();
        if (!rdoc.available()) return;
        if (renderdocCaptureOpen) {
            // Both axes can come due inside one still-open span. Say so: the second trigger has been
            // consumed and will not fire again, and a reader who set both deserves to know which one
            // actually opened the capture they got.
            std::fprintf(stderr, "[renderdoc] %s came due while a capture was already open; it is "
                                 "consumed without opening a second one\n", why);
            return;
        }
        renderdocDevicePointer = renderdoc_device_pointer();
        if (!renderdocDevicePointer)
            std::fprintf(stderr, "[renderdoc] WARNING: the live renderer has not published a Vulkan "
                                 "context yet -- capturing whatever device RenderDoc picks, which is "
                                 "usually the presentation device and contains NO guest draws\n");
        if (!rdoc.begin(renderdocDevicePointer)) return;
        renderdocCaptureOpen = true;
        renderdocOpenedAtGuestPresent = gpu::present_count();
        renderdocOpenIterations = 0;
        std::fprintf(stderr, "[renderdoc] capture opened by %s at guest present %llu\n", why,
                     (unsigned long long)renderdocOpenedAtGuestPresent);
    };
    bool scheduledScreenshotArmed = false;
    bool timedDumpPending = timedDumpMs > 0;
    bool running = true;
    bool paused = false;
    bool swapchainDirty = false;
    const SDL_WindowID appWindowId = SDL_GetWindowID(win);
    prosper::frontend::AppWindowControls windowControls;
    const SDL_WindowFlags initialWindowFlags = SDL_GetWindowFlags(win);
    bool fullscreenRequested = (initialWindowFlags & SDL_WINDOW_FULLSCREEN) != 0;
    windowControls.set_app_focus((initialWindowFlags & SDL_WINDOW_INPUT_FOCUS) != 0);

    // --fps. Brought up lazily, on the first frame that actually reaches the swapchain: at this
    // point the swapchain may not be final (the window can still be resized into fullscreen), and
    // the library view owns ImGui until a guest boots.
    //
    // Declared OUTSIDE the library-UI guard, because `Vk::overlay` and all five use sites are
    // outside it. Putting it inside compiled only because CMake defines that macro unconditionally
    // for this target, so the first person to make the library optional would get an undeclared type
    // in `struct Vk`.
    prosper::frontend::FpsOverlay fpsOverlay;
    vk.overlay = &fpsOverlay;
    // The HUD reports a ROLLING rate, not a run average: a title that ran well for a minute and then
    // collapsed must show the collapse. `fpsWindow` is re-based every kFpsWindowSeconds.
    constexpr double kFpsWindowSeconds = 1.0;
    prosper::gpu::PresentRateSnapshot fpsWindow = prosper::gpu::present_rate_snapshot();
    // The cumulative reading the window was closed against. Kept because "the picture is not
    // changing" cannot be reported honestly from a window alone: separating a static menu from a
    // title that has produced nothing is a question about the RUN (#3027). Held here rather than
    // re-snapshotted at each print site so the HUD and the stderr lines cannot disagree.
    prosper::gpu::PresentRateSnapshot fpsRun = fpsWindow;
    prosper::gpu::FrameRate fpsRate;
    std::vector<std::string> fpsLines;
    // The extent last presented through the GPU path, for the HUD's resolution field (#3010).
    uint32_t gpuPresentedW = 0, gpuPresentedH = 0;

#ifdef PROSPER_HAVE_LIBRARY_UI
    // The library replaces the empty idle window. Only meaningful when this run has no game of its own
    // and is not feeding a test pattern; a failure to bring it up is not fatal — the flat idle colour
    // remains, so a driver that cannot host the UI costs the user a library, not the app.
    prosper::frontend::LibraryUi libraryUi;
    std::string libraryStatus;
    // True while a picker opened from the library's own button is outstanding: its answer is a games
    // DIRECTORY to remember, not a title to boot.
    bool libraryBrowsePending = false;
    const bool wantLibrary = !testPattern && dump.empty();
    bool libraryHasGames = false;
    auto rescan_library = [&]() {
        if (!libraryUi.ready()) return;
        std::vector<prosper::frontend::GameEntry> found;
        if (!gamesDir.empty())
            found = prosper::frontend::scan_game_library(gamesDir, host_path_probe(), host_library_io());
        libraryHasGames = !found.empty();
        libraryUi.set_games(std::move(found), gamesDir);
    };
    if (wantLibrary) {
        libraryUi.set_music_preference(appConfig.launcher_music);
        if (libraryUi.init(win, vk.instance, vk.phys, vk.device, vk.qfamily, vk.queue, vk.swapchain,
                           vk.scFormat, vk.scImages, vk.scExtent)) {
            rescan_library();
            fprintf(stderr, "[app] library view ready.\n");
        } else {
            fprintf(stderr, "[app] library view unavailable; the window stays on the idle colour.\n");
        }
    }
#endif

    // Open a title the user handed the window — a dropped folder, or the folder picker's result
    // (#1469). argv never comes through here; it booted before the window existed.
    const prosper::frontend::GamePathProbe pathProbe = host_path_probe();
    // Per poll batch: at most one game is opened (a multi-folder drop must not start the first and
    // immediately switch away from it) and at most one "not a PS5 game" box is shown (dragging in a
    // folder of photos should not mean a dialog per file). Reset at the top of each batch.
    bool openedThisBatch = false;
    bool rejectedThisBatch = false;

    // Returns true when the path actually started something (booted here, or handed off to a new
    // process) — false when it was rejected or the start failed, so a caller can tell an ignored
    // path from a consumed one.
    auto open_game = [&](const std::string& picked) -> bool {
        const std::string root = prosper::frontend::resolve_app0_root(picked, pathProbe);
        switch (prosper::frontend::decide_open_action(root, g_boot_attempted)) {
        case prosper::frontend::GameOpenAction::ignore:
            fprintf(stderr, "[app] not a PS5 title: %s\n", picked.c_str());
            // The message box is modal and blocks this loop until dismissed. That is fine over an
            // idle window, but freezing a running game — including its guest dialog pump — over a
            // mis-aimed drag is not; there the log line is enough. One box per batch either way:
            // dragging a folder of photos in should not mean a dialog per file.
            if (!g_guest_started && !rejectedThisBatch) {
                rejectedThisBatch = true;
                SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_WARNING, "prosper",
                    ("That is not a PS5 game:\n\n" + picked +
                     "\n\nChoose the game's app0 folder -- the one holding eboot.bin and sce_sys.").c_str(),
                    win);
            }
            return false;
        case prosper::frontend::GameOpenAction::relaunch:
            // This process has spent its one boot (#352), so hand the new title to a fresh process
            // and let this one shut down normally.
            fprintf(stderr, "[app] this process has already booted; starting a new one for %s\n",
                    root.c_str());
            if (relaunch_with_dump(argc, argv, root)) {
                running = false;
                return true;
            }
            fprintf(stderr, "[app] could not start the new process\n");
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "prosper",
                "prosper runs one game per launch, and starting a second process failed.\n\n"
                "Quit and start prosper again with the other game.", win);
            return false;
        case prosper::frontend::GameOpenAction::boot_in_process: {
            fprintf(stderr, "[app] opening %s\n", root.c_str());
            // boot_program links and maps the whole module set inline, which takes seconds on a
            // large title and pumps no events meanwhile. Say so in the title bar first, or the
            // window just stops responding.
            SDL_SetWindowTitle(win, "prosper - loading...");
            std::string err;
            if (!start_guest(root, &err)) {
                fprintf(stderr, "[app] boot failed: %s\n", err.c_str());
                SDL_SetWindowTitle(win, title.c_str());
                SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "prosper",
                    ("Could not start that game.\n\n" + err +
                     "\n\nprosper gets one boot per launch, so open the next one from a fresh"
                     " start.").c_str(), win);
                return false;
            }
            title = window_title_for(root, false);
            SDL_SetWindowTitle(win, title.c_str());
            // A title opened after startup owns the captures from here on: name them after IT, not
            // after the empty window this process started as.
            {
                activeCaptureTitle = capture_title_for(root);
                grabNamer.set_title(activeCaptureTitle.id, activeCaptureTitle.label);
            }
#ifdef PROSPER_HAVE_LIBRARY_UI
            // Hand the swapchain back before the guest's first frame: from here the game present path
            // owns it, and two presenters acquiring the same images would fight.
            if (libraryUi.ready()) {
                fprintf(stderr, "[app] library view closed; presenting the game.\n");
                libraryUi.shutdown();
            }
#endif
            return true;
        }
        }
        return false;
    };

    // A launch that was not told what to show would otherwise be a dead end for anyone who did not
    // arrive through a command line.
    bool offerPicker = prosper::frontend::should_pick_at_startup(pick);
#ifdef PROSPER_HAVE_LIBRARY_UI
    // The library already shows a way in, so a modal dialog on top of it would be noise. Only fall back
    // to the picker when it has nothing to offer.
    if (offerPicker && libraryUi.ready() && libraryHasGames) {
        offerPicker = false;
        fprintf(stderr, "[app] library has titles; not opening the startup picker.\n");
    }
#endif
    if (offerPicker) {
        fprintf(stderr, "[app] no game given; opening the folder picker.\n");
        open_folder_picker(win);
    }

    // The automatic capture delay starts at app-loop entry, not process start or guest boot. This
    // gives unattended routes one explicit, repeatable host-time origin without desktop input.
    const uint64_t perfLoopStartNs = prosper::perf::monotonic_now_ns();
    while (running && !prosper_stop_requested()) {
        // Close a RenderDoc capture opened on the previous pass. Reporting the path is the whole
        // point of doing this here rather than firing and forgetting: an agent running headless has
        // no other way to learn where the capture went, and an abandoned capture (no API work
        // recorded in the span) returns an empty path, which is said out loud rather than implied
        // by a missing file.
        if (renderdocCaptureOpen) {
            const uint64_t guestNow = gpu::present_count();
            const bool guestPresented =
                guestNow - renderdocOpenedAtGuestPresent >= renderdocPresents;
            const bool hitCap = ++renderdocOpenIterations >= renderdocMaxIterations;
            if (guestPresented || hitCap) {
                renderdocCaptureOpen = false;
                const std::string rdocPath =
                    prosper::frontend::RenderDocCapture::instance().end(renderdocDevicePointer);
                // Report WHY the span closed. A capture closed by the cap rather than by a guest
                // present is a different artifact -- it may hold no guest frame at all -- and a
                // reader who cannot tell the two apart will read an empty capture as an empty frame.
                const char* closedBy = guestPresented ? "guest present" : "iteration cap";
                if (rdocPath.empty())
                    std::fprintf(stderr,
                                 "[renderdoc] capture ABANDONED after %llu iterations (closed by "
                                 "%s) -- RenderDoc recorded no API work in the span\n",
                                 (unsigned long long)renderdocOpenIterations, closedBy);
                else {
                    // Warning first, for the same reason as the shutdown note below.
                    char note[512];
                    std::snprintf(note, sizeof note,
                                  "%sprosper capture. Closed by %s after %llu app-loop iteration(s) "
                                  "and %llu guest present(s). Aimed at %s.",
                                  guestPresented ? ""
                                                 : "TRUNCATED -- closed by the iteration cap, not "
                                                   "by a guest present: this may hold only part of "
                                                   "a frame, or no guest frame at all. ",
                                  closedBy, (unsigned long long)renderdocOpenIterations,
                                  (unsigned long long)(guestNow - renderdocOpenedAtGuestPresent),
                                  renderdocDevicePointer ? "the live renderer's Vulkan device"
                                                         : "whichever device RenderDoc chose");
                    prosper::frontend::RenderDocCapture::instance().set_comments(rdocPath, note);
                    std::fprintf(stderr,
                                 "[renderdoc] capture written: %s (spanned %llu app-loop iterations "
                                 "and %llu guest presents, closed by %s)\n",
                                 rdocPath.c_str(), (unsigned long long)renderdocOpenIterations,
                                 (unsigned long long)(guestNow - renderdocOpenedAtGuestPresent),
                                 closedBy);
                }
            }
        }
        // F8's only always-on work is this 4 Hz sample. `sample_due` is one atomic read, so the
        // process CPU/RSS query is not paid on every UI-loop iteration.
        const uint64_t perfNow = prosper::perf::monotonic_now_ns();
        if (perfCapture.sample_due(perfNow)) {
            perfCapture.observe_sample(prosper::perf::collect_process_sample(
                perfNow, gpu::present_count(),
                prosper::frontend::rendered_frame_counter(
                    vk.gpu_present, gpu::present_frame_seq()),
                shown));
        }
        prosper::perf::CaptureOutcome perfOutcome;
        if (perfCapture.take_outcome(perfOutcome)) {
            if (perfOutcome.ok) {
                std::fprintf(stderr,
                    "[perf] capture written (pre=%zu post=%zu renderer=%zu compute=%zu "
                    "dropped=%zu/%zu) -> %s\n",
                    perfOutcome.pre_samples, perfOutcome.post_samples,
                    perfOutcome.renderer_records, perfOutcome.compute_records,
                    perfOutcome.renderer_dropped, perfOutcome.compute_dropped,
                    perfOutcome.path.c_str());
            } else {
                std::fprintf(stderr, "[perf] %s capture FAILED: %s\n",
                             perfCaptureWasAutomatic ? "automatic" : "F8",
                             perfOutcome.error.c_str());
            }
            perfCaptureWasAutomatic = false;
        }
        // Consume a completion before starting a new automatic capture. Otherwise a user-triggered
        // F8 outcome that completed on this same sample could be mislabeled as automatic after the
        // new arm replaces the trigger-provenance flag.
        if (automaticPerfCapture.take_if_due(perfLoopStartNs, perfNow)) {
            arm_performance_capture(
                true, perfNow, (perfNow - perfLoopStartNs) / 1'000'000);
        }
        // Same one-shot contract as the perf trigger above: the trigger is consumed before the arm is
        // attempted, so a failed arm stays visible instead of silently retrying into a later phase.
        if (automaticGrabAfter.take_if_due(perfLoopStartNs, perfNow))
            arm_frame_grab(true, "PROSPER_GRAB_BUNDLE_AFTER_MS");
        if (automaticRdocAfter.take_if_due(perfLoopStartNs, perfNow))
            arm_renderdoc_capture("PROSPER_RENDERDOC_AFTER_MS");
        openedThisBatch = rejectedThisBatch = false;
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
#ifdef PROSPER_HAVE_LIBRARY_UI
            // Only while the library owns the screen: once a guest boots, every key belongs to it.
            if (libraryUi.ready() && !g_guest_started && libraryUi.handle_event(ev)) continue;
#endif
            if (ev.type == SDL_EVENT_QUIT) running = false;
            else if (ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
                     ev.window.windowID == appWindowId) {
                // A progress utility window means the app window is not necessarily SDL's last
                // window, so closing it no longer guarantees a synthesized SDL_EVENT_QUIT.
                running = false;
            } else if (ev.type == SDL_EVENT_KEY_DOWN || ev.type == SDL_EVENT_KEY_UP) {
                prosper::frontend::AppWindowKey key{};
                key.app_window = ev.key.windowID == appWindowId;
                key.pressed = ev.key.down;
                key.repeat = ev.key.repeat;
                key.escape = ev.key.key == SDLK_ESCAPE;
                key.pause = ev.key.key == SDLK_PAUSE || ev.key.key == SDLK_F10;
                key.f11 = ev.key.key == SDLK_F11;
                key.enter = ev.key.key == SDLK_RETURN || ev.key.key == SDLK_KP_ENTER;
                key.alt = (ev.key.mod & SDL_KMOD_ALT) != 0;
                // Interactive performance capture: unlike F9 this retains no GPU commands, pixels,
                // or ordinary frame dumps. It freezes the cheap rolling counter ring and enables
                // structured renderer/compute timing only for the five seconds after this press.
                // F8 is host-owned and is not forwarded to the guest IME/pad path.
                if (ev.type == SDL_EVENT_KEY_DOWN && key.app_window && !ev.key.repeat &&
                    ev.key.key == SDLK_F8) {
                    const uint64_t armedAt = prosper::perf::monotonic_now_ns();
                    arm_performance_capture(false, armedAt);
                    continue;
                }
                // Interactive frame grab: F9 arms a one-shot capture of the next COMPLETE frame (every
                // submit between the next two presents) into a replayable .prgbundle, plus a screenshot.
                // The whole-frame bundle re-runs the frame's producer submits on replay, so renderer-owned
                // RTTs regenerate instead of replaying black (as a single-submit .prgcap does for a
                // deferred renderer). It is a HOST hotkey, NOT forwarded to the guest IME/pad. On-demand
                // only — near-zero cost until pressed, so it never distorts the FPS you are observing.
                if (ev.type == SDL_EVENT_KEY_DOWN && key.app_window && !ev.key.repeat &&
                    ev.key.key == SDLK_F9) {
                    arm_frame_grab(false, "hotkey");
                    continue;
                }
                // F6 / F7: author a render snapshot from the frame currently on screen. F6 means
                // "this looks CORRECT", F7 means "this looks WRONG" -- and the wrong ones are kept
                // deliberately, because a title with no guard improves silently. Both are HOST
                // hotkeys and are not forwarded to the guest. Armed here, written when the next
                // frame is presented, so the pixels are the ones that were actually on screen.
                if (ev.type == SDL_EVENT_KEY_DOWN && key.app_window && !ev.key.repeat &&
                    (ev.key.key == SDLK_F6 || ev.key.key == SDLK_F7)) {
                    if (pendingSnapVerdict) {
                        std::fprintf(stderr,
                                     "[snap] a snap is already armed and waiting for the next "
                                     "presented frame; this press is ignored\n");
                    } else {
                        pendingSnapVerdict = ev.key.key == SDLK_F6
                            ? prosper::frontend::SnapVerdict::correct
                            : prosper::frontend::SnapVerdict::incorrect;
                        // SHIFT marks the frame as one to SCAN for rather than expect at a fixed
                        // offset -- for anything sitting after a loading screen, an FMV, or any
                        // other stretch whose length depends on the machine. The person pressing
                        // the key is the only one who knows that, and they know it right now.
                        pendingSnapMode = (ev.key.mod & SDL_KMOD_SHIFT)
                            ? prosper::frontend::SnapMode::scan
                            : prosper::frontend::SnapMode::anchor;
                    }
                    continue;
                }
                // Ctrl+O opens a title in the empty window (#1469). Deliberately unavailable once a
                // game is running: the guest owns the keyboard from that point (O is its R1), and a
                // second title needs a new process anyway — drop a folder on the window for that.
                // Excluded under --test-pattern too, matching should_pick_at_startup: that run has a
                // producer feeding the present layer already, and a guest would fight it.
                if (ev.type == SDL_EVENT_KEY_DOWN && key.app_window && !ev.key.repeat &&
                    ev.key.key == SDLK_O && (ev.key.mod & SDL_KMOD_CTRL) &&
                    !g_guest_started && !testPattern) {
                    open_folder_picker(win);
                    continue;
                }
                // #1093: forward app-window keys to the guest's IME keyboard path. Titles like
                // PPSA02664 read input through sceImeUpdate, not libScePad. SDL3 scancodes ARE USB
                // HID usage ids for the keyboard page — exactly the keycode the guest event wants.
                // Deliver clean press/release edges (skip auto-repeat).
                if (key.app_window && !ev.key.repeat && !key.pause)
                    prosper::ime_push_key((uint16_t)ev.key.scancode, ev.key.down);
                switch (windowControls.handle_key(key)) {
                case prosper::frontend::AppWindowCommand::quit:
                    running = false;
                    break;
                case prosper::frontend::AppWindowCommand::toggle_pause:
                    paused = !paused;
                    if (paused) {
                        // Close the producer gate before freezing queued device audio.
                        prosper_set_paused(true);
#ifdef PROSPER_AUDIO_SDL3
                        prosper::set_sdl3_audio_paused(true);
#endif
                    } else {
                        // Start the device before releasing producers so the first resumed grain
                        // cannot queue behind a device that is still paused.
#ifdef PROSPER_AUDIO_SDL3
                        prosper::set_sdl3_audio_paused(false);
#endif
                        prosper_set_paused(false);
                    }
                    SDL_SetWindowTitle(win, paused ? (title + " - paused").c_str() : title.c_str());
                    fprintf(stderr, "[app] %s at guest flip boundary\n",
                            paused ? "pause requested" : "resumed");
                    break;
                case prosper::frontend::AppWindowCommand::toggle_fullscreen: {
                    // SDL may apply fullscreen requests asynchronously. Toggle the last accepted
                    // target instead of reading a flag that can still describe the old state.
                    const bool targetFullscreen = !fullscreenRequested;
                    if (!SDL_SetWindowFullscreen(win, targetFullscreen)) {
                        fprintf(stderr, "[app] fullscreen toggle failed: %s\n", SDL_GetError());
                    } else {
                        fullscreenRequested = targetFullscreen;
                        swapchainDirty = true;
                        fprintf(stderr, "[app] fullscreen %s requested\n",
                                targetFullscreen ? "on" : "off");
                    }
                    break;
                }
                case prosper::frontend::AppWindowCommand::none:
                    break;
                }
            } else if (ev.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED &&
                       ev.window.windowID == appWindowId) {
                swapchainDirty = true;
            } else if (ev.type == SDL_EVENT_WINDOW_FOCUS_LOST &&
                       ev.window.windowID == appWindowId) {
                windowControls.set_app_focus(false);
            } else if (ev.type == SDL_EVENT_WINDOW_FOCUS_GAINED &&
                       ev.window.windowID == appWindowId) {
                windowControls.set_app_focus(true);
            } else if (ev.type == SDL_EVENT_WINDOW_ENTER_FULLSCREEN &&
                       ev.window.windowID == appWindowId) {
                fullscreenRequested = true;
            } else if (ev.type == SDL_EVENT_WINDOW_LEAVE_FULLSCREEN &&
                       ev.window.windowID == appWindowId) {
                fullscreenRequested = false;
            } else if (ev.type == SDL_EVENT_DROP_FILE && ev.drop.windowID == appWindowId) {
                // A game folder dropped on the window (#1469). Works whether or not one is running:
                // with a game up this relaunches, which is the only way to switch titles today.
                // Excluded under --test-pattern, which already has a frame producer.
                // SDL emits one event per dropped URI, so a multi-folder drop must not start the
                // first and immediately relaunch into the second — `openedThisBatch` takes only the
                // first one that lands. A REJECTED drop does not consume the batch, so a stray file
                // alongside a real title is harmless.
                if (ev.drop.data && !openedThisBatch && !testPattern)
                    openedThisBatch = open_game(ev.drop.data);
            }
        }
        if (!running) break;

        // The folder picker answers asynchronously and possibly on another thread, so it parks its
        // result. Consume it here, between frames, where booting is safe.
        //
        // Discard the answer if a game started while the dialog was open — on the zenity backend the
        // dialog is a separate process and the window stays interactive, so a folder can be dropped
        // meanwhile; honouring the dialog afterwards would relaunch straight back out of the game it
        // just started. A *failed* boot deliberately does not discard: no game is running, and
        // picking again is exactly how the user recovers (it relaunches into a fresh process).
        {
            std::string picked;
            bool pickerStillOpen = false;
            // One critical section: the callback takes the same mutex, so it runs strictly before or
            // strictly after this, and the "empty path but closed dialog" state cannot be observed
            // while an answer is in flight.
            {
                std::lock_guard<std::mutex> lock(g_picked_mutex);
                picked.swap(g_picked_path);
                pickerStillOpen = g_picker_open;
            }
#ifdef PROSPER_HAVE_LIBRARY_UI
            // The dialog closed without an answer (cancelled, or it failed to open): disarm, or the
            // flag would still be set when some later, unrelated pick arrives.
            if (libraryBrowsePending && picked.empty() && !pickerStillOpen)
                libraryBrowsePending = false;
            // The library asked for this folder, so its answer names a games DIRECTORY to remember, not
            // a title to boot. Without this the result went to open_game(), which resolves an app0 root
            // and therefore always rejected a folder-of-folders with "That is not a PS5 game".
            if (!picked.empty() && libraryBrowsePending) {
                libraryBrowsePending = false;
                if (!host_path_probe().is_dir(picked)) {
                    libraryStatus = "That is not a folder.";
                } else {
                    prosper::frontend::AppConfig cfg = load_app_config();
                    cfg.games_dir = prosper::frontend::strip_trailing_separators(picked);
                    gamesDir = cfg.games_dir;
                    libraryStatus = save_app_config(cfg) ? std::string()
                                                        : "Could not save the games folder setting.";
                    fprintf(stderr, "[app] games directory set to %s\n", gamesDir.c_str());
                    rescan_library();
                }
                picked.clear();
            }
#endif
            if (!picked.empty() && !testPattern) {
                if (g_guest_started)
                    fprintf(stderr, "[app] ignoring the folder picker's answer: a game started first.\n");
                else
                    open_game(picked);
            }
            if (!running) break;
        }

        // Snapshot input before any minimized-window early exit. This clears released guest
        // buttons even while swapchain recreation has to wait for a non-zero pixel extent.
        const bool* keyboard = SDL_GetKeyboardState(nullptr);
        const bool enterDown = keyboard[SDL_SCANCODE_RETURN] ||
                               keyboard[SDL_SCANCODE_RETURN2] ||
                               keyboard[SDL_SCANCODE_KP_ENTER];
        windowControls.reconcile_enter(enterDown);
        poll_keyboard(keyboard, windowControls.guest_options_allowed());
#ifdef PROSPER_HAVE_DIALOG_SDL3
        prosper::sdl_platform_ui_pump();   // run a pending ImeDialog text-entry modal on this (main) thread
#endif

        if (swapchainDirty) {
            int dw = 0, dh = 0;
            SDL_GetWindowSizeInPixels(win, &dw, &dh);
            if (dw <= 0 || dh <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;   // minimized or between fullscreen modes; retry after the next event
            }
            // SDL's non-zero pixel size can lead the Win32 Vulkan surface during a display/DPI
            // transition. Do not destroy the still-valid swapchain until the surface publishes a
            // usable extent too; create_swapchain would otherwise see 0x0 and the app would exit
            // after an ordinary transient resize.
            VkSurfaceCapabilitiesKHR resizeCaps{};
            const VkResult resizeCapsResult = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                vk.phys, vk.surface, &resizeCaps);
            if (resizeCapsResult != VK_SUCCESS) {
                fprintf(stderr, "[app] Vulkan error %d while querying a resized surface\n",
                        static_cast<int>(resizeCapsResult));
                running = false;
                break;
            }
            if (!prosper::frontend::swapchain_extent_available(
                    resizeCaps.currentExtent,
                    {static_cast<uint32_t>(dw), static_cast<uint32_t>(dh)})) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            {
                // #1270: on a shared present queue, hold the submit mutex across the device drain so no
                // guest submit races vkDeviceWaitIdle (which waits on all queues). Only the wait needs it;
                // the swapchain destroy/create below do not touch the render queue.
                std::unique_lock<std::mutex> lk(gpu::shared_present_submit_mutex(), std::defer_lock);
                if (vk.gpu_present && vk.queue_shared) lk.lock();
                vkDeviceWaitIdle(vk.device);
            }
            if (vk.gpu_present && gpuPrevSlot >= 0) {
                prosper::frontend::present_blit_release(gpuPrevSlot); gpuPrevSlot = -1;
            }
            if (vk.swapchain) vkDestroySwapchainKHR(vk.device, vk.swapchain, nullptr);
            vk.swapchain = VK_NULL_HANDLE;
            if (!create_swapchain(vk, static_cast<uint32_t>(dw), static_cast<uint32_t>(dh),
                                  requestedPresentMode)) {
                fprintf(stderr, "[app] could not recreate the swapchain after a window-size change\n");
                running = false;
                break;
            }
            swapchainDirty = false;
#ifdef PROSPER_HAVE_LIBRARY_UI
            if (libraryUi.ready() &&
                !libraryUi.recreate_swapchain(vk.swapchain, vk.scFormat, vk.scImages, vk.scExtent)) {
                fprintf(stderr, "[app] library view lost its swapchain; falling back to the idle colour.\n");
                libraryUi.shutdown();
            }
#endif
            // The framebuffers point at the destroyed swapchain's images. Rebuilt, or the HUD turns
            // itself off -- never left pointing at freed images.
            // On failure the HUD tears itself down but `showFps` stays set, so the `if (showFps)`
            // block below rebuilds it from scratch on the next iteration -- which is the behaviour
            // we want (a resize should not cost the counter for the rest of the run). Say that,
            // rather than "now off", which the next iteration usually makes untrue.
            if (fpsOverlay.ready() &&
                !fpsOverlay.recreate_swapchain(vk.scFormat, vk.scImages, vk.scExtent))
                fprintf(stderr, "[app] the fps overlay lost its swapchain; rebuilding it.\n");
        }
        const auto loopNow = std::chrono::steady_clock::now();
        if (timedDumpPending && loopNow >= nextTimedDump) {
            ++timedDumpCount;
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                loopNow - loopStarted).count();
            fprintf(stderr, "[app] timed guest-state dump #%u after %lld ms at frame %llu\n",
                    timedDumpCount, (long long)elapsed, (unsigned long long)shown);
            if (timedDumpCount == 1) dump_guest_exception_trace();
            dump_guest_thread_trace(timedDumpPath, timedDumpPthread);
            // #2139: what the threads were DOING before they all stopped. Empty unless
            // PROSPER_SYNC_RING is set, so this costs nothing on an ordinary run.
            prosper::dump_guest_sync_trace(timedDumpPath);
            if (timedDumpIntervalMs > 0)
                nextTimedDump = loopNow + std::chrono::milliseconds(timedDumpIntervalMs);
            else
                timedDumpPending = false;
        }

        // Arm the next authored anchor once the guest reaches it. This sits in the COMMON path,
        // deliberately: presentation has a GPU route and a CPU-fallback route, and the headless
        // offscreen driver used by automated checks takes the fallback -- arming inside the GPU
        // branch made the trigger fire on a window and do nothing at all in CI, which is precisely
        // the case it exists for.
        //
        // Forward-only: every target at or below the current ordinal is consumed in one pass, so a
        // stall that skips several anchors captures the frame that actually exists rather than
        // replaying the backlog against later scenes.
        if (nextSnapTarget < snapFlipTargets.size() && !pendingActualTarget) {
            const int64_t nowFlip = prosper_pad_flip_ordinal();
            if (nowFlip >= 0) {
                while (nextSnapTarget + 1 < snapFlipTargets.size() &&
                       snapFlipTargets[nextSnapTarget + 1] <= nowFlip)
                    ++nextSnapTarget;
                if (snapFlipTargets[nextSnapTarget] <= nowFlip)
                    pendingActualTarget = snapFlipTargets[nextSnapTarget++];
            }
        }

        // No game yet and no test pattern: nothing is producing frames. Draw the library so the window
        // is a way IN rather than a dead end (#1471), and fall back to a flat colour when the UI is
        // unavailable so the window is never left on undefined swapchain contents (#1469).
        // present_frame_seq() counts every frame handed to the present layer, so the moment anything
        // publishes one — a game opened here, or an external producer — the real path takes over.
        // These frames deliberately do not count toward --frames: that gate asserts real content.
        if (!testPattern && !g_guest_started && gpu::present_frame_seq() == 0) {
#ifdef PROSPER_HAVE_LIBRARY_UI
            if (libraryUi.ready()) {
                const prosper::frontend::LibraryAction act = libraryUi.render_frame(libraryStatus);
                if (libraryUi.needs_recreate()) {
                    libraryUi.clear_needs_recreate();
                    swapchainDirty = true;   // serviced at the top of the next iteration
                }
                switch (act.kind) {
                case prosper::frontend::LibraryAction::Kind::open:
                    // Straight through the same start_guest() path argv, the drop target and the
                    // picker all use — the library chooses a title, it does not boot one differently.
                    if (!open_game(act.app0_root)) libraryStatus = "Could not start that game.";
                    break;
                case prosper::frontend::LibraryAction::Kind::browse:
                    // Arm only when this request is the one that opened the dialog: otherwise a browse
                    // raised while another picker is already up would claim that picker's answer.
                    if (open_folder_picker(win)) libraryBrowsePending = true;
                    break;
                case prosper::frontend::LibraryAction::Kind::quit:
                    running = false;
                    break;
                case prosper::frontend::LibraryAction::Kind::set_music: {
                    // Persist the choice, keeping every other setting (including ones this build does
                    // not understand) intact. A write failure is not worth interrupting the user over;
                    // the toggle still applies for this run.
                    prosper::frontend::AppConfig cfg = load_app_config();
                    cfg.launcher_music = act.music_on;
                    if (!save_app_config(cfg))
                        fprintf(stderr, "[app] could not persist the music setting\n");
                    break;
                }
                case prosper::frontend::LibraryAction::Kind::set_games_dir:
                case prosper::frontend::LibraryAction::Kind::none:
                    break;
                }
                if (!running) break;
                continue;
            }
#endif
            static const uint8_t kIdlePixel[4] = {0x1e, 0x1e, 0x1e, 0xff};
            static uint8_t kIdleFrame[4 * 4 * 4];
            for (size_t i = 0; i < sizeof kIdleFrame; i++) kIdleFrame[i] = kIdlePixel[i % 4];
            if (present_frame(vk, kIdleFrame, 4, 4) == PresentAttempt::out_of_date)
                swapchainDirty = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            continue;
        }

        static const uint32_t kPatW = 1920, kPatH = 1080;
        if (testPattern && !paused) feed_test_pattern(kPatW, kPatH, patFrame++);

        // Cheap calibration companion to a heavyweight F9 bundle. Read back exactly one presented
        // frame, with no GPU command capture, so long routes can locate a visual checkpoint before
        // scheduling the replayable bundle there. Do not overwrite an interactive F9 screenshot;
        // if both coincide, the scheduled shot remains eligible for the following real present.
        // Frame-addressed captures reuse the screenshot's due rule, so a skipped swapchain ordinal
        // leaves the one-shot eligible for the next real present rather than missing silently.
        if (prosper::frontend::capture_frame_due(scheduledGrabFrame, shown, scheduledGrabArmed)) {
            scheduledGrabArmed = true;
            arm_frame_grab(true, "PROSPER_GRAB_BUNDLE_AT_FRAME");
        }
        // RenderDoc's capture spans a frame explicitly, so unlike F8/F9 it needs a matching end.
        // Start it here, with its siblings, and close it at the top of the next iteration: that
        // span covers one whole app-loop pass, which is the unit a reader means by "this frame".
        if (prosper::frontend::capture_frame_due(scheduledRdocFrame, shown, scheduledRdocArmed)) {
            scheduledRdocArmed = true;
            arm_renderdoc_capture("PROSPER_RENDERDOC_AT_FRAME");
        }
        if (prosper::frontend::capture_frame_due(scheduledPerfFrame, shown, scheduledPerfArmed)) {
            scheduledPerfArmed = true;
            // Report the REAL elapsed time, not a placeholder: this line is read as a measurement,
            // and "fired at 0 ms" for a capture armed 40 s in is simply false.
            arm_performance_capture(true, perfNow,
                                    (perfNow - perfLoopStartNs) / 1'000'000);
        }
        if (pendingGrabScreenshot.empty() && prosper::frontend::capture_frame_due(
                scheduledScreenshotFrame, shown, scheduledScreenshotArmed)) {
            pendingGrabScreenshot = scheduledScreenshotPath;
            pendingGrabGuestPresent = gpu::present_count();
            // Both halves of "what is pending" are set here: an explicitly configured path is not a
            // claimed frame-grab name, so it has no suffix and this frontend does not own it.
            pendingGrabSuffix = 0;
            pendingGrabReserved = false;
            scheduledScreenshotArmed = true;
            fprintf(stderr,
                    "[grab] scheduled screenshot armed at host frame %llu (guest present %llu)\n",
                    static_cast<unsigned long long>(shown + 1),
                    static_cast<unsigned long long>(pendingGrabGuestPresent));
        }

        // Present the latest finished frame from the core's present layer.
        // The frame dimensions: from the guest's registered VideoOut display in normal use; in
        // test-pattern mode there is no guest, so use the dims we feed (present_width/height report
        // the VideoOut registry, which is empty without a guest). Either way, readback needs a
        // buffer sized to the frame it holds — guard zero dims so we never present a 0-extent image.
        // #1587: the user pressed a key. Report what the grab actually did, in the window title as
        // well as stderr — a silent failure is indistinguishable from a keystroke that never
        // registered, which is the conclusion several agents reached on a 4K title. The title notice
        // is deliberately non-modal: F9 can be pressed during a routed run, and a message box would
        // block the render loop mid-capture.
        {
            prosper::gpu::InteractiveGrabOutcome grab;
            if (prosper::gpu::take_interactive_grab_outcome(grab)) {
                unsigned bundleSuffix = 0;
                // Not every outcome is a frame grab's: the same channel reports the env-driven
                // captures, whose path this frontend never created.
                const bool reservedHere = take_bundle_reservation(grab.bundle_path, bundleSuffix);
                if (grab.ok) {
                    // The file exists now, so this line may name it.
                    std::fprintf(stderr, "%s\n",
                                 prosper::frontend::frame_grab_write_line(
                                     "bundle", grab.bundle_path, bundleSuffix).c_str());
                    grabNotice = "F9: captured " +
                        std::filesystem::path(grab.bundle_path).filename().string();
                } else {
                    // What became of the output path. Every branch reports the state it actually
                    // found, and NOTHING is deleted unless this frontend created it: a capture
                    // configured through PROSPER_CAPTURE_BUNDLE* arrives on this same channel with a
                    // path prosper never reserved, and a zero-byte file sitting at such a path
                    // belongs to whoever put it there.
                    std::string state;
                    if (grab.bundle_path.empty()) {
                        state = "no output path was recorded for it";
                    } else if (!reservedHere) {
                        state = "this path was configured, not reserved by a frame grab: nothing was "
                                "written to it and nothing was removed";
                    } else {
                        std::error_code ec;
                        const bool present = std::filesystem::exists(grab.bundle_path, ec);
                        if (ec) {
                            // "I could not tell" is not "gone" — an unreadable parent directory
                            // lands here, and naming the wrong cause is what this logging avoids.
                            state = "its reserved file could not be checked: " + ec.message();
                        } else if (!present) {
                            state = "its reserved file is already gone";
                        } else {
                            const uintmax_t bytes = std::filesystem::file_size(grab.bundle_path, ec);
                            if (ec) {
                                state = "its reserved file could not be examined: " + ec.message();
                            } else if (bytes != 0) {
                                // Cannot happen on this path today; if it ever does, the file has real
                                // content and is not ours to delete.
                                state = "its reserved file holds " + std::to_string(bytes) +
                                        " bytes and was left alone";
                            } else {
                                // Leaving a zero-byte .prgbundle after saying none was written would
                                // be the same class of untrue statement this logging exists to avoid.
                                const bool removed = std::filesystem::remove(grab.bundle_path, ec);
                                if (removed && !ec) state = "the empty reservation has been removed";
                                // remove() reports false with NO error when the file was already
                                // gone — a race with an external deleter. Printing "could not be
                                // removed: Success" there would assert a failure that did not happen.
                                else if (!ec)       state = "the empty reservation was already gone";
                                else                state = "the empty reservation could not be removed: " +
                                                            ec.message();
                            }
                        }
                    }
                    std::fprintf(stderr,
                        "\n=========================== F9 FRAME GRAB FAILED ===========================\n"
                        "  %s\n"
                        "  no bundle content was written; the output path was %s\n"
                        "  %s\n"
                        "  %s\n"
                        "  budget in force: %llu MiB -- raise it with "
                        "PROSPER_CAPTURE_BUNDLE_MAX_MB=<64..3072> and re-arm the capture\n"
                        "============================================================================\n\n",
                        grab.error.c_str(), grab.bundle_path.c_str(), state.c_str(),
                        reservedHere
                            ? "the .bmp screenshot for this press is a separate file with the same stem"
                            : "this capture was configured, not pressed; it has no screenshot",
                        static_cast<unsigned long long>(grab.max_unique_bytes >> 20));
                    grabNotice = "F9 GRAB FAILED - see console (raise PROSPER_CAPTURE_BUNDLE_MAX_MB)";
                }
                grabNoticeUntil = std::chrono::steady_clock::now() + std::chrono::seconds(grab.ok ? 4 : 12);
                if (win) SDL_SetWindowTitle(win, (title + " - " + grabNotice).c_str());
            }
            if (!grabNotice.empty() && std::chrono::steady_clock::now() >= grabNoticeUntil) {
                grabNotice.clear();
                if (win) SDL_SetWindowTitle(win, title.c_str());
            }
        }

        // --fps: bring the HUD up on the first iteration that will actually present a game frame,
        // and refresh the rolling rate at most once per window. Deferred to here rather than done at
        // startup because the library view owns ImGui until a guest boots, and because the swapchain
        // is not final until the window has settled.
        if (showFps) {
            if (!fpsOverlay.ready()) {
                if (!fpsOverlay.init(vk.instance, vk.phys, vk.device, vk.qfamily, vk.queue,
                                     vk.scFormat, vk.scImages, vk.scExtent)) {
                    fprintf(stderr, "[app] --fps could not start; continuing without the counter.\n");
                    showFps = false;
                } else if (vk.gpu_present) {
                    // GPU present publishes no CPU pixels, so the counter has to sample the
                    // presented image itself (#3010). Allocated only here: with --fps off the
                    // present path stays byte-for-byte what it was.
                    ensure_fps_sample(vk);
                }
            }

            const prosper::gpu::PresentRateSnapshot now = prosper::gpu::present_rate_snapshot();
            if (prosper::frontend::fps_window_due(fpsWindow.now_seconds, now.now_seconds,
                                                  kFpsWindowSeconds)) {
                fpsRate = prosper::gpu::frame_rate_between(fpsWindow, now);
                fpsWindow = now;
                fpsRun = now;
                // present_frame_width/height are fed by the CPU publish path and read 0 under GPU
                // present, so the HUD showed "0x0" there. Prefer the extent actually presented.
                fpsLines = prosper::frontend::fps_hud_lines(
                    fpsRate,
                    gpuPresentedW ? gpuPresentedW : gpu::present_frame_width(),
                    gpuPresentedH ? gpuPresentedH : gpu::present_frame_height(),
                    now);
            }
        }
        // Only a HUD that has something to say is passed down; an empty list leaves both present
        // paths on their original barrier.
        const std::vector<std::string>* fpsForPresent =
            (showFps && fpsOverlay.ready() && !fpsLines.empty()) ? &fpsLines : nullptr;

        // Render completion and guest flips are separate clocks: the command stream can flip before
        // the renderer publishes its CPU frame. Key this loop to the completed-frame sequence so a
        // late renderer publication is not missed or marked handled while only the previous frame exists.
        if (vk.gpu_present) {
            // GPU present (#1270): blit the renderer's front-buffer image straight to the swapchain.
            prosper::frontend::GpuScanoutFrame gf;
            if (prosper::frontend::present_blit_acquire(gf)) {
                if (!prosper::frontend::present_source_is_newer(
                        havePresentedGuestFlip, lastPresentedGuestFlip, gf.frame_seq)) {
                    prosper::frontend::present_blit_release(gf.slot);
                    continue;
                }
                bool grabReady = false;
                // A pending snap needs the presented pixels just as a grab screenshot does, so it
                // must also request the readback -- without this the GPU path never stages them and
                // F6/F7 would silently do nothing on the fast path.
                PresentAttempt attempt = present_frame_gpu(
                    vk, gf, gpuPrevSlot,
                    !pendingGrabScreenshot.empty() || pendingSnapVerdict.has_value() ||
                        pendingActualTarget.has_value(), grabReady,
                    showFps, fpsForPresent);
                gpuPresentedW = gf.width; gpuPresentedH = gf.height;
                if (grabReady) {
                    flushGrabScreenshot(static_cast<const uint8_t*>(vk.stageMapped),
                                        gf.width, gf.height);
                    flushPendingSnap(static_cast<const uint8_t*>(vk.stageMapped),
                                     gf.width, gf.height);
                    flushPendingActual(static_cast<const uint8_t*>(vk.stageMapped),
                                       gf.width, gf.height);
                }
                if (attempt == PresentAttempt::out_of_date) {
                    swapchainDirty = true;
                } else if (attempt == PresentAttempt::skipped) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(4));
                } else if (attempt == PresentAttempt::failed) {
                    running = false;
                } else {
                    shown.record(prosper::frontend::PresentedFrameSource::GpuScanout, running);
                    lastFrameProgress = std::chrono::steady_clock::now();
                    havePresentedGuestFlip = true;
                    lastPresentedGuestFlip = gf.frame_seq;
                    static auto t0 = std::chrono::steady_clock::now(); static uint64_t mark = 0;
                    if (shown - mark >= 60) {
                        auto now = std::chrono::steady_clock::now();
                        double s = std::chrono::duration<double>(now - t0).count();
                        // The PRESENTED rate, and -- when --fps armed the content sample (#3010) --
                        // the DISTINCT rate beside it. Both, because they answer different questions:
                        // a title whose picture is frozen still presents at 60, and this line alone
                        // read healthy right through the Windows splash stall in #3011.
                        //
                        // The "picture not changing" verdict is classified against `fpsRun` -- the
                        // cumulative snapshot -- and not against the one-second window.
                        // present_frame_rate.hpp requires the run's active share beside that verdict,
                        // and a differenced window cannot supply it; this line used to lean on its own
                        // presented rate to carry that job, which was an accident of the formatting
                        // rather than anything the contract could check (#3027).
                        std::string distinct;
                        if (showFps && fpsRate.measured) {
                            char measured[48];
                            snprintf(measured, sizeof measured, ", %.1f distinct",
                                     fpsRate.distinct_fps);
                            distinct = measured;
                            const std::string note = prosper::gpu::format_unchanged_picture(
                                prosper::gpu::unchanged_picture(fpsRate, fpsRun));
                            if (!note.empty()) distinct += " -- " + note;
                        }
                        fprintf(stderr, "[app] %.1f fps (%llu frames, gpu-present%s)\n",
                                (shown - mark) / (s > 0 ? s : 1), (unsigned long long)shown,
                                distinct.c_str());

                        t0 = now; mark = shown;
                    }
                }
            } else if (gpu::present_frame_seq() != lastFrameSeq) {
                // #1270 Finding 2: no GPU frame was published this iteration. On a publish MISS (front
                // target evicted/invalidated, or no free slot) the renderer still did the CPU readback, so
                // present that CPU frame rather than stranding the window on a stale/black image. Also
                // covers startup before the first publish. present_frame serializes on the shared queue.
                gpu::PresentFrameLease cf;
                if (gpu::present_acquire_rendered_frame(cf) && cf.width && cf.height && cf.rgba &&
                    cf.rgba->size() == (size_t)cf.width * cf.height * 4) {
                    if (!prosper::frontend::present_source_is_newer(
                            havePresentedGuestFlip, lastPresentedGuestFlip,
                            cf.guest_present_count)) {
                        // Consume this CPU publication even though it is stale. Otherwise every
                        // GPU acquire gap would redisplay the same old fallback indefinitely.
                        lastFrameSeq = cf.frame_seq;
                        continue;
                    }
                    PresentAttempt a = present_frame(vk, cf.rgba->data(), cf.width, cf.height,
                                                     fpsForPresent);
                    if (gpuPrevSlot >= 0) { prosper::frontend::present_blit_release(gpuPrevSlot); gpuPrevSlot = -1; }
                    if (a == PresentAttempt::out_of_date) swapchainDirty = true;
                    else if (a == PresentAttempt::skipped) std::this_thread::sleep_for(std::chrono::milliseconds(4));
                    else if (a == PresentAttempt::failed) running = false;
                    else {
                        lastFrameSeq = cf.frame_seq;
                        shown.record(
                            prosper::frontend::PresentedFrameSource::GpuCpuFallback, running);
                        lastFrameProgress = std::chrono::steady_clock::now();
                        havePresentedGuestFlip = true;
                        lastPresentedGuestFlip = cf.guest_present_count;
                        flushGrabScreenshot(cf.rgba->data(), cf.width, cf.height);
                        flushPendingSnap(cf.rgba->data(), cf.width, cf.height);
                        flushPendingActual(cf.rgba->data(), cf.width, cf.height);
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));   // no new GPU or CPU frame yet
            }
        } else {
        bool newFrame = testPattern || (gpu::present_frame_seq() != lastFrameSeq);
        gpu::PresentFrameLease frame;
        if (newFrame && gpu::present_acquire_rendered_frame(frame)) {
            uint32_t w = frame.width;
            uint32_t h = frame.height;
            if (w == 0 || h == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }
            if (frame.rgba && frame.rgba->size() == (size_t)w * h * 4) {
                PresentAttempt attempt = present_frame(vk, frame.rgba->data(), w, h, fpsForPresent);
                if (attempt == PresentAttempt::out_of_date) {
                    // Out-of-date/suboptimal: share the resize/fullscreen recreation path next loop.
                    swapchainDirty = true;
                } else if (attempt == PresentAttempt::skipped) {
                    // Window occluded/minimized (#1182): no swapchain image within the bounded acquire.
                    // Leave lastFrameSeq unchanged so we present the freshest frame once the window is
                    // visible again, and do not mark the swapchain dirty. The guest keeps running.
                    std::this_thread::sleep_for(std::chrono::milliseconds(4));
                } else if (attempt == PresentAttempt::failed) {
                    running = false;
                } else {
                    lastFrameSeq = frame.frame_seq;
                    shown.record(prosper::frontend::PresentedFrameSource::Cpu, running);
                    lastFrameProgress = std::chrono::steady_clock::now();
                    flushGrabScreenshot(frame.rgba->data(), w, h);
                    flushPendingSnap(frame.rgba->data(), w, h);
                    flushPendingActual(frame.rgba->data(), w, h);
                    // Periodic present-rate log (every 60 presented frames).
                    static auto t0 = std::chrono::steady_clock::now(); static uint64_t mark = 0;
                    if (shown - mark >= 60) {
                        auto now = std::chrono::steady_clock::now();
                        double s = std::chrono::duration<double>(now - t0).count();
                        if (frameTrace) {
                            size_t nonzeroRgbBytes = 0;
                            for (size_t i = 0; i + 3 < frame.rgba->size(); i += 4) {
                                nonzeroRgbBytes += (*frame.rgba)[i] != 0;
                                nonzeroRgbBytes += (*frame.rgba)[i + 1] != 0;
                                nonzeroRgbBytes += (*frame.rgba)[i + 2] != 0;
                            }
                            fprintf(stderr,
                                    "[app] %.1f fps (%llu frames) render_seq=%llu flips=%llu "
                                    "nonzero_rgb_bytes=%zu/%zu\n",
                                    (shown - mark) / (s > 0 ? s : 1),
                                    (unsigned long long)shown,
                                    (unsigned long long)frame.frame_seq,
                                    (unsigned long long)gpu::present_count(), nonzeroRgbBytes,
                                    (size_t)w * h * 3);
                        } else {
                            // Same two rates as the GPU branch (#3010), so a run can be compared
                            // across present paths. Here the distinct rate comes from the CPU
                            // publish hash rather than from a sampled grid. Same run-classified
                            // verdict too, for the reason given in that branch (#3027).
                            std::string distinct;
                            if (showFps && fpsRate.measured) {
                                char measured[48];
                                snprintf(measured, sizeof measured, ", %.1f distinct",
                                         fpsRate.distinct_fps);
                                distinct = measured;
                                const std::string note = prosper::gpu::format_unchanged_picture(
                                    prosper::gpu::unchanged_picture(fpsRate, fpsRun));
                                if (!note.empty()) distinct += " -- " + note;
                            }
                            fprintf(stderr, "[app] %.1f fps (%llu frames%s)\n",
                                    (shown - mark) / (s > 0 ? s : 1),
                                    (unsigned long long)shown, distinct.c_str());
                        }

                        t0 = now; mark = shown;
                    }
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));   // no new frame — don't spin
        }
        }   // end CPU-present branch (#1270)
        if (stallDumpMs > 0 &&
            std::chrono::steady_clock::now() - lastFrameProgress >=
                std::chrono::milliseconds(stallDumpMs)) {
            fprintf(stderr, "[app] no presented-frame progress for %d ms at frame %llu\n",
                    stallDumpMs, (unsigned long long)shown);
            dump_guest_exception_trace();
            lastFrameProgress = std::chrono::steady_clock::now();
        }
    }

    // B1 (#3322 review): the span closes at the TOP of the loop, so every exit path -- --frames
    // satisfied, SDL_EVENT_QUIT, or the elapsed trigger firing shortly before a title exits -- left
    // an armed capture open and the artifact was lost with no line saying so. That is precisely the
    // silent-instrument failure this trigger exists to prevent, so it is closed here too.
    //
    // KNOWN LIMIT, not covered here (#3324): the guest can leave the process without ever returning
    // to this line. hle_kernel_time.cpp's k_exit (guest _exit) and k_debug_raise_release (UE4's
    // shipping check()) call _Exit from the GUEST thread while main is still inside the loop, and
    // _Exit is exit_group -- so no atexit hook would help either. A capture armed on a title that
    // dies that way is still lost silently. Both sites already flush diagnostics before their
    // deliberate _Exit, so a registered flush list is the natural home; that is a cross-layer seam
    // which does not exist yet and is deliberately not invented here.
    //
    // Unlike its F8 sibling below, this CLOSES rather than cancels. RenderDoc decides for itself
    // whether the span held anything: a span containing real work is written and worth having even
    // though it ended at shutdown rather than on a guest present, and one containing nothing returns
    // 0 and reports ABANDONED. Either way the run says what happened, which cancelling could not.
    if (renderdocCaptureOpen) {
        renderdocCaptureOpen = false;
        const std::string rdocPath =
            prosper::frontend::RenderDocCapture::instance().end(renderdocDevicePointer);
        if (rdocPath.empty())
            std::fprintf(stderr, "[renderdoc] capture ABANDONED at shutdown after %llu app-loop "
                                 "iteration(s) -- RenderDoc recorded no API work in the span\n",
                         (unsigned long long)renderdocOpenIterations);
        else {
            // The warning goes IN THE FILE, not only in stderr (#3322 re-review). A truncated
            // capture opens perfectly and reads as a whole frame with draws missing -- the same
            // shape as the wrong-device capture this trigger exists to prevent -- and the stderr
            // line does not travel with the artifact.
            // The warning leads. If these strings ever grow past the buffer, snprintf truncates
            // the TAIL -- so the safety-critical clause must not sit there, or a future edit
            // silently produces a confident-looking stamp with the caveat cut off.
            char note[512];
            std::snprintf(note, sizeof note,
                          "TRUNCATED -- missing draws are expected here and are NOT evidence that "
                          "the guest failed to submit them. prosper capture, closed by application "
                          "SHUTDOWN after %llu app-loop iteration(s), NOT by a guest present, so it "
                          "may hold only part of a frame. Aimed at %s.",
                          (unsigned long long)renderdocOpenIterations,
                          renderdocDevicePointer ? "the live renderer's Vulkan device"
                                                 : "whichever device RenderDoc chose");
            prosper::frontend::RenderDocCapture::instance().set_comments(rdocPath, note);
            std::fprintf(stderr, "[renderdoc] capture written: %s (closed by SHUTDOWN after %llu "
                                 "app-loop iteration(s), NOT by a guest present -- it may hold only "
                                 "part of a frame; the file itself is stamped with that warning)\n",
                         rdocPath.c_str(), (unsigned long long)renderdocOpenIterations);
        }
    }
    perfCapture.cancel();     // never publish a short/incomplete .prperf on a graceful early exit
    prosper_request_stop();   // signal the guest run-loop to wind down at its next boundary
    fprintf(stderr, "[app] shutting down after %llu presented frame(s)\n", (unsigned long long)shown);
    const int exitCode = (exitAfter && (int)shown < exitAfter) ? 1 : 0;

    // run_entry does not yet observe the frontend stop flag, so a booted guest cannot be joined.
    // Returning from main after detaching it is unsafe: C++ static teardown destroys HLE/renderer
    // state while guest threads still use it (a short --frames run reliably ended in 0xC0000005 on
    // Windows). Until the flip-boundary cooperative stop is implemented, terminate directly and
    // let the OS reclaim process state without running destructors under the live guest.
    if (g_guest_thread.joinable()) {
        g_guest_thread.detach();

        // #3225: bound the window in which the detached guest thread can be inside a GPU submission
        // when std::_Exit fires. _Exit becomes exit_group(), and a thread that is inside an amdgpu
        // command submission at that moment cannot be torn down until it returns from the kernel —
        // it parks in __drm_exec_lock_obj on a GEM reservation nobody will now release, the process
        // becomes an unreapable zombie, and the compositor's own DRM work blocks behind it. That
        // froze the developer's desktop twice in one day, recoverable only by a root-forced GPU
        // reset.
        //
        // Closing the gate stops NEW guest submissions (they return VK_ERROR_DEVICE_LOST, which
        // every submission site already handles as "not submitted"), which is what lets the drain
        // reach zero instead of chasing a moving count. This BOUNDS the window; it does not close
        // it. A thread already inside vkQueueSubmit is waited for, and if it never comes out the
        // drain times out and we exit exactly as before. The real fix is the cooperative stop the
        // comment above names as unimplemented — observe the flag, stop at a submission boundary,
        // and JOIN.
        prosper::gpu_submit_gate_begin_shutdown();
        constexpr int kGuestSubmitDrainMs = 2000;
        if (prosper::gpu_submit_gate_drain(kGuestSubmitDrainMs)) {
            // Only after a SUCCESSFUL drain. vkDeviceWaitIdle requires host access to every VkQueue
            // of the device to be externally synchronized, and the drain is what provides that here
            // (the main loop uses the shared-present submit mutex for the same reason at the
            // swapchain-resize wait above). A timed-out drain means a guest submit may still be
            // live, so waiting would both violate that and be the very block we are avoiding.
            //
            // Note what this wait does and does not cost. It is UNBOUNDED — on a title that has
            // hung a GPU job it returns only when the amdgpu watchdog resets the device, adding
            // seconds to a close — and it is not protecting a teardown, because the _Exit below
            // destroys no Vulkan object. What it buys is that already-submitted guest work is off
            // the device before the process dies. That is a deliberate trade of a bounded, visible
            // delay against the unbounded, invisible one this whole path exists to prevent; if it
            // ever becomes the bigger nuisance, deleting it is safe and loses only that guarantee.
            if (vk.device) vkDeviceWaitIdle(vk.device);
        } else {
            fprintf(stderr,
                    "[app] guest GPU submissions did not drain within %d ms (%d still in flight); "
                    "exiting anyway\n",
                    kGuestSubmitDrainMs, prosper::gpu_submit_gate_in_flight());
        }

#ifdef PROSPER_HAVE_LIVE_RENDERER
        // _Exit skips RuntimeComputeTimingSelector's destructor. Publish its validity verdict here;
        // the report is idempotent so a future cooperative teardown cannot duplicate it.
        prosper::frontend::report_live_compute_timing_selector_summary();
#endif
        // The bounded dmem writer trace also has a destructor/atexit fallback, which _Exit skips.
        prosper::host::guest_dmem_write_trace_report();
        fflush(nullptr);
        std::_Exit(exitCode);
    }

    // Before SDL_Quit, like the library view: this releases ImGui and Vulkan objects while the
    // device and the window still exist, and the destructor would do it too late. Note this runs
    // BEFORE the vkDeviceWaitIdle below and must not depend on it -- FpsOverlay::shutdown() drains
    // the device itself, exactly as LibraryUi::shutdown() does (library_ui.cpp:288), so the
    // guarantee belongs to the object rather than to the order of these four lines.
    fpsOverlay.shutdown();
#ifdef PROSPER_HAVE_LIBRARY_UI
    // Before SDL_Quit: ImGui's SDL3 backend frees cursors and closes gamepads on shutdown, and the
    // destructor would otherwise run after SDL had already torn those down.
    libraryUi.shutdown();
#endif
    vkDeviceWaitIdle(vk.device);
    // After the idle above: these are read by the present command buffer, so they can only be
    // released once no submit can still be reading them (#3010).
    if (vk.sampleMapped) vkUnmapMemory(vk.device, vk.sampleMem);
    if (vk.sampleImg)    vkDestroyImage(vk.device, vk.sampleImg, nullptr);
    if (vk.sampleImgMem) vkFreeMemory(vk.device, vk.sampleImgMem, nullptr);
    if (vk.sampleBuf)    vkDestroyBuffer(vk.device, vk.sampleBuf, nullptr);
    if (vk.sampleMem)    vkFreeMemory(vk.device, vk.sampleMem, nullptr);

    SDL_DestroyWindow(win); SDL_Quit();
    return exitCode;
}
