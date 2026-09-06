/* control.c — a known-answer control for the pixel-history tool.
 *
 * THE PROBLEM THIS SOLVES. `pixel_history.py` reports why a pixel ended up the colour
 * it did: nothing drew there, something drew and was rejected (and by which test), the
 * shader ran and computed black, or the shader computed a colour and the store lost it.
 * Those four answers steer completely different investigations, so the tool is only
 * worth having if the distinctions are real. Nothing in a game capture can establish
 * that -- the game is the unknown. So this program constructs each case BY HAND, at a
 * named pixel, and the tool's report is checked against what was built.
 *
 * THE CONSTRUCTION. One 64x64 target, cleared to RED, then five full-screen draws that
 * differ in exactly ONE property each. At the probe pixel (32,32):
 *
 *   draw 1  green   depth 0.5  full scissor        -> PASSES, pixel becomes green
 *   draw 2  blue    depth 0.9  full scissor        -> DEPTH TEST FAILS (0.9 >= 0.5)
 *   draw 3  white   depth 0.1  scissor {0,0,4,4}   -> SCISSORED OUT (0.1 would have won)
 *   draw 4  magenta depth 0.1  full scissor, kill  -> SHADER DISCARDS
 *   draw 5  yellow  depth 0.2  full scissor        -> PASSES, pixel becomes yellow
 *
 * Draw 3 is deliberately the draw that WOULD have won on depth. If the scissor were not
 * applied it would write white and leave depth at 0.1, and draw 5 would then fail -- so
 * the final colour discriminates "scissor worked" from "scissor ignored" without needing
 * the tool at all. Draw 4 discards a fragment that would otherwise have passed, so
 * "discarded" cannot be confused with "never reached".
 *
 * SELF-CHECK. The program reads the pixel back and asserts yellow BEFORE any tool looks
 * at the capture. A control that is itself broken must fail here rather than silently
 * become the thing the tool is validated against.
 *
 * Usage: pixel_history_control                 (self-check only, no capture)
 *        pixel_history_control <path template> (built with PROSPER_PIXHIST_CAPTURE)
 * Exit: 0 the construction rendered as designed, 1 it did not, 2 setup/tool error.
 */

#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM      64u
#define PROBE_X  32u
#define PROBE_Y  32u

static const uint32_t kVertSpv[] = {
#include "control_vert_spv.h"
};
static const uint32_t kFragSpv[] = {
#include "control_frag_spv.h"
};

static VkInstance g_instance;
static VkPhysicalDevice g_pd;
static VkDevice g_dev;
static VkQueue g_queue;
static uint32_t g_queue_family;

static void die(const char *what, VkResult r)
{
    fprintf(stderr, "pixel_history_control: %s failed (%d)\n", what, (int)r);
    exit(2);
}

static void require(int cond, const char *what)
{
    if (!cond) {
        fprintf(stderr, "pixel_history_control: %s\n", what);
        exit(2);
    }
}

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

/* Push constants only: no descriptor set, so a draw's identity is entirely its
 * push constant + scissor. That is what makes single-property attribution possible. */
struct push {
    float    color[4];
    float    depth;
    uint32_t discard;
};

struct arm {
    const char *name;
    struct push pc;
    int         narrow_scissor;
    const char *expect;
};

#ifdef PROSPER_PIXHIST_CAPTURE
#include "../doctor/renderdoc_control.h"   /* reuses the merged capture bracket verbatim */
int main(int argc, char **argv)
#else
int main(void)
#endif
{
    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef PROSPER_PIXHIST_CAPTURE
    require(argc == 2, "usage: pixel_history_control <new capture path template>");
#endif

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "pixel_history_control";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkResult r = vkCreateInstance(&ici, NULL, &g_instance);
    if (r) die("vkCreateInstance", r);

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(g_instance, &ndev, NULL);
    require(ndev > 0, "no Vulkan physical device");
    require(ndev <= 8, "too many devices");
    VkPhysicalDevice devs[8];
    vkEnumeratePhysicalDevices(g_instance, &ndev, devs);
    g_pd = devs[0];

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL);
    require(nq > 0 && nq <= 16, "bad queue family count");
    VkQueueFamilyProperties qf[16];
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qf);
    g_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i)
        if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { g_queue_family = i; break; }
    require(g_queue_family != UINT32_MAX, "no graphics queue family");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    r = vkCreateDevice(g_pd, &dci, NULL, &g_dev);
    if (r) die("vkCreateDevice", r);
    vkGetDeviceQueue(g_dev, g_queue_family, 0, &g_queue);
#ifdef PROSPER_PIXHIST_CAPTURE
    capture_begin(argv[1]);
#endif

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_pd, &props);
    printf("device: %s\n", props.deviceName);

    /* ---- colour + depth targets ---------------------------------------------------- */
    VkImage color_img, depth_img;
    VkDeviceMemory color_mem, depth_mem;
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent.width = DIM; ii.extent.height = DIM; ii.extent.depth = 1;
    ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    r = vkCreateImage(g_dev, &ii, NULL, &color_img);
    if (r) die("vkCreateImage(color)", r);

    VkImageCreateInfo di = ii;
    di.format = VK_FORMAT_D32_SFLOAT;
    di.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    r = vkCreateImage(g_dev, &di, NULL, &depth_img);
    if (r) die("vkCreateImage(depth)", r);

    VkMemoryRequirements mr;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    vkGetImageMemoryRequirements(g_dev, color_img, &mr);
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    require(ai.memoryTypeIndex != UINT32_MAX, "no device-local memory type (color)");
    r = vkAllocateMemory(g_dev, &ai, NULL, &color_mem);
    if (r) die("vkAllocateMemory(color)", r);
    vkBindImageMemory(g_dev, color_img, color_mem, 0);

    vkGetImageMemoryRequirements(g_dev, depth_img, &mr);
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    require(ai.memoryTypeIndex != UINT32_MAX, "no device-local memory type (depth)");
    r = vkAllocateMemory(g_dev, &ai, NULL, &depth_mem);
    if (r) die("vkAllocateMemory(depth)", r);
    vkBindImageMemory(g_dev, depth_img, depth_mem, 0);

    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    VkImageView color_view, depth_view;
    vi.image = color_img; vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r = vkCreateImageView(g_dev, &vi, NULL, &color_view);
    if (r) die("vkCreateImageView(color)", r);
    vi.image = depth_img; vi.format = VK_FORMAT_D32_SFLOAT;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    r = vkCreateImageView(g_dev, &vi, NULL, &depth_view);
    if (r) die("vkCreateImageView(depth)", r);

    /* ---- render pass ---------------------------------------------------------------- */
    VkAttachmentDescription att[2] = {{0}, {0}};
    att[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    att[0].samples = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    att[1] = att[0];
    att[1].format = VK_FORMAT_D32_SFLOAT;
    att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference cref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference dref = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub = {0};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &cref;
    sub.pDepthStencilAttachment = &dref;
    VkRenderPassCreateInfo rpi = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 2; rpi.pAttachments = att;
    rpi.subpassCount = 1; rpi.pSubpasses = &sub;
    VkRenderPass pass;
    r = vkCreateRenderPass(g_dev, &rpi, NULL, &pass);
    if (r) die("vkCreateRenderPass", r);

    VkImageView views[2] = {color_view, depth_view};
    VkFramebufferCreateInfo fi = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = pass; fi.attachmentCount = 2; fi.pAttachments = views;
    fi.width = DIM; fi.height = DIM; fi.layers = 1;
    VkFramebuffer fb;
    r = vkCreateFramebuffer(g_dev, &fi, NULL, &fb);
    if (r) die("vkCreateFramebuffer", r);

    /* ---- pipeline ------------------------------------------------------------------- */
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    VkShaderModule vs, fs;
    smi.codeSize = sizeof(kVertSpv); smi.pCode = kVertSpv;
    r = vkCreateShaderModule(g_dev, &smi, NULL, &vs);
    if (r) die("vkCreateShaderModule(vs)", r);
    smi.codeSize = sizeof(kFragSpv); smi.pCode = kFragSpv;
    r = vkCreateShaderModule(g_dev, &smi, NULL, &fs);
    if (r) die("vkCreateShaderModule(fs)", r);

    VkPushConstantRange pcr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(struct push)};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout layout;
    r = vkCreatePipelineLayout(g_dev, &pli, NULL, &layout);
    if (r) die("vkCreatePipelineLayout", r);

    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, 
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs; stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vin = {
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo iasm = {
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    iasm.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp = {0.0f, 0.0f, (float)DIM, (float)DIM, 0.0f, 1.0f};
    VkRect2D full = {{0, 0}, {DIM, DIM}};
    VkPipelineViewportStateCreateInfo vps = {
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vps.viewportCount = 1; vps.pViewports = &vp;
    vps.scissorCount = 1; vps.pScissors = &full;
    VkPipelineRasterizationStateCreateInfo rs = {
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms = {
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds = {
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState cba = {0};
    cba.colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo cb = {
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &cba;
    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn = {
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 1; dyn.pDynamicStates = dyn_states;

    VkGraphicsPipelineCreateInfo gpi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.stageCount = 2; gpi.pStages = stages;
    gpi.pVertexInputState = &vin;
    gpi.pInputAssemblyState = &iasm;
    gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pDepthStencilState = &ds;
    gpi.pColorBlendState = &cb;
    gpi.pDynamicState = &dyn;
    gpi.layout = layout;
    gpi.renderPass = pass;
    VkPipeline pipeline;
    r = vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gpi, NULL, &pipeline);
    if (r) die("vkCreateGraphicsPipelines", r);

    /* ---- readback buffer ------------------------------------------------------------ */
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = (VkDeviceSize)DIM * DIM * 4;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer readback;
    r = vkCreateBuffer(g_dev, &bci, NULL, &readback);
    if (r) die("vkCreateBuffer", r);
    vkGetBufferMemoryRequirements(g_dev, readback, &mr);
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memory_type(mr.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    require(ai.memoryTypeIndex != UINT32_MAX, "no host-visible memory type");
    VkDeviceMemory readback_mem;
    r = vkAllocateMemory(g_dev, &ai, NULL, &readback_mem);
    if (r) die("vkAllocateMemory(readback)", r);
    vkBindBufferMemory(g_dev, readback, readback_mem, 0);

    /* ---- the five arms -------------------------------------------------------------- */
    const struct arm arms[5] = {
        {"green  depth 0.5",       {{0,1,0,1}, 0.5f, 0}, 0, "PASS"},
        {"blue   depth 0.9",       {{0,0,1,1}, 0.9f, 0}, 0, "depthTestFailed"},
        {"white  depth 0.1 sciss", {{1,1,1,1}, 0.1f, 0}, 1, "scissorClipped"},
        {"magent depth 0.1 kill",  {{1,0,1,1}, 0.1f, 1}, 0, "shaderDiscarded"},
        {"yellow depth 0.2",       {{1,1,0,1}, 0.2f, 0}, 0, "PASS"},
    };

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = g_queue_family;
    VkCommandPool cpool;
    r = vkCreateCommandPool(g_dev, &cpi, NULL, &cpool);
    if (r) die("vkCreateCommandPool", r);
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cpool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    r = vkAllocateCommandBuffers(g_dev, &cai, &cmd);
    if (r) die("vkAllocateCommandBuffers", r);

    VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbi);

    VkClearValue clears[2];
    clears[0].color.float32[0] = 1.0f; clears[0].color.float32[1] = 0.0f;
    clears[0].color.float32[2] = 0.0f; clears[0].color.float32[3] = 1.0f;  /* RED */
    clears[1].depthStencil.depth = 1.0f;
    clears[1].depthStencil.stencil = 0;
    VkRenderPassBeginInfo rbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = pass; rbi.framebuffer = fb;
    rbi.renderArea = full;
    rbi.clearValueCount = 2; rbi.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkRect2D narrow = {{0, 0}, {4, 4}};
    for (int i = 0; i < 5; ++i) {
        vkCmdSetScissor(cmd, 0, 1, arms[i].narrow_scissor ? &narrow : &full);
        vkCmdPushConstants(cmd, layout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(struct push), &arms[i].pc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
    vkCmdEndRenderPass(cmd);

    VkImageMemoryBarrier imb = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    imb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    imb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imb.image = color_img;
    imb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imb.subresourceRange.levelCount = 1;
    imb.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &imb);

    VkBufferImageCopy region = {0};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = DIM;
    region.imageExtent.height = DIM;
    region.imageExtent.depth = 1;
    vkCmdCopyImageToBuffer(cmd, color_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback, 1, &region);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    r = vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE);
    if (r) die("vkQueueSubmit", r);
    r = vkQueueWaitIdle(g_queue);
    if (r) die("vkQueueWaitIdle", r);

    /* ---- self-check: the construction must have rendered as designed ---------------- */
    void *mapped = NULL;
    r = vkMapMemory(g_dev, readback_mem, 0, VK_WHOLE_SIZE, 0, &mapped);
    if (r) die("vkMapMemory", r);
    const unsigned char *px = (const unsigned char *)mapped + (PROBE_Y * DIM + PROBE_X) * 4;
    const unsigned char want[4] = {255, 255, 0, 255};   /* yellow: draw 5 won */
    int ok = memcmp(px, want, 4) == 0;

    printf("probe pixel (%u,%u) = %u,%u,%u,%u (expected %u,%u,%u,%u)\n",
           PROBE_X, PROBE_Y, px[0], px[1], px[2], px[3],
           want[0], want[1], want[2], want[3]);
    for (int i = 0; i < 5; ++i)
        printf("  arm %d  %-24s expect %s\n", i + 1, arms[i].name, arms[i].expect);
    printf("PIXEL_HISTORY_CONTROL=%s\n", ok ? "CONSTRUCTED" : "MISRENDERED");
    if (!ok)
        fprintf(stderr, "the control did not render as designed; do not validate against it\n");

    vkUnmapMemory(g_dev, readback_mem);
    vkDeviceWaitIdle(g_dev);
#ifdef PROSPER_PIXHIST_CAPTURE
    capture_end();
#endif
    return ok ? 0 : 1;
}
