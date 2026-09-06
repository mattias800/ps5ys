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
 * THE CONSTRUCTION. One 64x64 target, transfer-cleared to BLACK before the render pass
 * (a transfer clear, deliberately, so it is not itself a pixel-history event), then draws
 * confined by scissor to five regions. Each region probes a DIFFERENT verdict, because a
 * control that only ever produces one of them tests the machinery and not the distinctions:
 *
 *   A  (16,16)  five arms, below           -> PIXEL_WAS_WRITTEN, final YELLOW
 *   A' (16,36)  arm 1 only draws here      -> final GREEN; this is what proves arm 1 ran
 *   B  (48,16)  one passing draw, black    -> SHADER_WROTE_BLACK
 *   C  (16,48)  one passing draw, bright,
 *               through a colorWriteMask=0
 *               pipeline                   -> STORE_LOST_IT, final BLACK
 *   E  (48,48)  one draw, discarded        -> ALL_REJECTED (with the clear present)
 *
 * Region A's arms, in submission order, at (16,16):
 *
 *   1 green   depth 0.5  wide scissor  -> PASSES  (also paints A', which is how it is seen)
 *   2 white   depth 0.1  tiny scissor  -> SCISSORED OUT (0.1 would have won)
 *   3 magenta depth 0.1  kill          -> SHADER DISCARDS
 *   4 yellow  depth 0.2                -> PASSES, pixel becomes yellow
 *   5 blue    depth 0.9                -> DEPTH TEST FAILS, and it is LAST on purpose
 *
 * Arm 5 is last so that a broken depth test is visible in the final colour: if it wrongly
 * passed, the pixel would end blue. In the first version of this control the depth-fail arm
 * ran second, and a broken depth test was invisible -- the later lower-depth arm overwrote
 * it either way. Arm 2 is the draw that would otherwise have won on depth, so an ignored
 * scissor ends the pixel white; arm 3 discards a fragment that would otherwise have passed,
 * so "discarded" cannot be confused with "never reached".
 *
 * WHAT THE SELF-CHECK ESTABLISHES, AND WHAT IT DOES NOT. Reading five pixels back proves
 * the construction reached the intended FINAL STATE in every region. It does not prove each
 * arm was rejected for the intended REASON -- a scissor and a discard both leave the pixel
 * untouched. That is the tool's job: `pixel_history.py --expect-control` compares the whole
 * per-event outcome list, and it is the check that fails if a rejection is misattributed.
 * Do not read a green self-check as validating the tool.
 *
 * NOTHING_DREW is not constructed here. Every scissored draw still produces a pixel-history
 * entry at pixels outside its rectangle, so no pixel of this target has an empty history.
 * That verdict is exercised separately, against a capture whose geometry misses the pixel.
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
    VkRect2D    scissor;
    int         no_color_write;   /* selects the colorWriteMask=0 pipeline */
    const char *expect;
};

struct probe {
    const char   *region;
    uint32_t      x, y;
    unsigned char want[4];
    const char   *verdict;
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
    /* LOAD, not CLEAR: a render-pass clear is itself a pixel-history event that passes at
     * every pixel, which would make ALL_REJECTED unconstructible. The black ground comes
     * from a transfer clear before the pass instead. */
    att[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    att[1] = att[0];
    att[1].format = VK_FORMAT_D32_SFLOAT;
    att[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;    /* depth still starts at 1.0 */
    att[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
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

    /* Identical but for the write mask. A draw through this passes every test and computes a
     * colour the target never receives -- which is exactly STORE_LOST_IT, and is the verdict
     * the first version of this control could not construct at all. */
    VkPipelineColorBlendAttachmentState cba0 = cba;
    cba0.colorWriteMask = 0;
    VkPipelineColorBlendStateCreateInfo cb0 = cb;
    cb0.pAttachments = &cba0;
    gpi.pColorBlendState = &cb0;
    VkPipeline pipeline_no_write;
    r = vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gpi, NULL, &pipeline_no_write);
    if (r) die("vkCreateGraphicsPipelines(no-write)", r);

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

    /* ---- the regions ---------------------------------------------------------------- */
    const VkRect2D A      = {{0, 0}, {32, 32}};    /* the sequence                        */
    const VkRect2D A_wide = {{0, 0}, {32, 40}};    /* A plus the strip only arm 1 paints  */
    const VkRect2D A_tiny = {{0, 0}, {4, 4}};      /* the scissor arm 2 is confined to    */
    const VkRect2D B      = {{32, 0}, {32, 32}};
    const VkRect2D C      = {{0, 40}, {32, 24}};
    const VkRect2D E      = {{32, 32}, {32, 32}};

    const struct arm arms[8] = {
        {"A1 green  0.5 wide", {{0, 1, 0, 1}, 0.5f, 0}, A_wide, 0, "PASS"},
        {"A2 white  0.1 tiny", {{1, 1, 1, 1}, 0.1f, 0}, A_tiny, 0, "scissorClipped at A"},
        {"A3 magent 0.1 kill", {{1, 0, 1, 1}, 0.1f, 1}, A,      0, "shaderDiscarded"},
        {"A4 yellow 0.2",      {{1, 1, 0, 1}, 0.2f, 0}, A,      0, "PASS"},
        {"A5 blue   0.9 LAST", {{0, 0, 1, 1}, 0.9f, 0}, A,      0, "depthTestFailed"},
        {"B  black  0.5",      {{0, 0, 0, 1}, 0.5f, 0}, B,      0, "PASS, computes black"},
        {"C  bright 0.5 cwm0", {{0, 0.8f, 0.9f, 1}, 0.5f, 0}, C, 1, "PASS, write mask drops it"},
        {"E  kill   0.5",      {{1, 0.5f, 0, 1}, 0.5f, 1}, E,   0, "shaderDiscarded"},
    };

    /* What each region must look like afterwards, and the verdict it exists to construct. */
    const struct probe probes[5] = {
        {"A  sequence",   16, 16, {255, 255,   0, 255}, "PIXEL_WAS_WRITTEN"},
        {"A' arm-1 only", 16, 36, {  0, 255,   0, 255}, "(proves arm 1 rendered)"},
        {"B  black draw", 48, 16, {  0,   0,   0, 255}, "SHADER_WROTE_BLACK"},
        {"C  no write",   16, 48, {  0,   0,   0, 255}, "STORE_LOST_IT"},
        {"E  all killed", 48, 48, {  0,   0,   0, 255}, "ALL_REJECTED"},
    };

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = g_queue_family;
    VkCommandPool cpool;
    r = vkCreateCommandPool(g_dev, &cpi, NULL, &cpool);
    if (r) die("vkCreateCommandPool", r);
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cpool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 2;
    VkCommandBuffer cmds[2];
    r = vkAllocateCommandBuffers(g_dev, &cai, cmds);
    if (r) die("vkAllocateCommandBuffers", r);
    VkCommandBuffer cmd = cmds[0];

#ifdef PROSPER_PIXHIST_CAPTURE
    capture_begin(argv[1]);
#endif
    VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbi);

    /* Black ground by transfer clear, INSIDE the capture on purpose.
     *
     * MEASURED, not assumed: RenderDoc's Vulkan pixel history reports vkCmdClearColorImage
     * as a PASSING modification, with no test evaluated (trap 269). That briefly made
     * region E read SHADER_WROTE_BLACK -- the clear was its only passing event.
     *
     * The first fix moved the clear out of the capture, which made the control pass by
     * removing the phenomenon from it. That is backwards: the tool has to survive this on
     * every real frame, since every real frame clears its targets. So the clear stays in,
     * classify() distinguishes clears from draws, and this control now FAILS if that
     * distinction is ever lost. A control that avoids the hard case guards nothing. */
    VkImageMemoryBarrier pre = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    pre.srcAccessMask = 0;
    pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    pre.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre.image = color_img;
    pre.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    pre.subresourceRange.levelCount = 1;
    pre.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &pre);
    VkClearColorValue black = {{0.0f, 0.0f, 0.0f, 1.0f}};
    vkCmdClearColorImage(cmd, color_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black,
                         1, &pre.subresourceRange);
    pre.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    pre.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL,
                         1, &pre);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo ground = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    ground.commandBufferCount = 1; ground.pCommandBuffers = &cmd;
    r = vkQueueSubmit(g_queue, 1, &ground, VK_NULL_HANDLE);
    if (r) die("vkQueueSubmit(ground)", r);
    r = vkQueueWaitIdle(g_queue);
    if (r) die("vkQueueWaitIdle(ground)", r);

    cmd = cmds[1];
    vkBeginCommandBuffer(cmd, &cbi);

    VkClearValue clears[2];
    clears[0].color.float32[0] = 0.0f; clears[0].color.float32[1] = 0.0f;
    clears[0].color.float32[2] = 0.0f; clears[0].color.float32[3] = 1.0f;
    clears[1].depthStencil.depth = 1.0f;
    clears[1].depthStencil.stencil = 0;
    VkRenderPassBeginInfo rbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = pass; rbi.framebuffer = fb;
    rbi.renderArea = full;
    rbi.clearValueCount = 2; rbi.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    for (int i = 0; i < 8; ++i) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          arms[i].no_color_write ? pipeline_no_write : pipeline);
        vkCmdSetScissor(cmd, 0, 1, &arms[i].scissor);
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

    /* ---- self-check: every region must have reached its intended final state ---------- */
    void *mapped = NULL;
    r = vkMapMemory(g_dev, readback_mem, 0, VK_WHOLE_SIZE, 0, &mapped);
    if (r) die("vkMapMemory", r);
    int ok = 1;
    for (int i = 0; i < 5; ++i) {
        const unsigned char *px =
            (const unsigned char *)mapped + (probes[i].y * DIM + probes[i].x) * 4;
        int good = memcmp(px, probes[i].want, 4) == 0;
        ok &= good;
        printf("  %-14s (%2u,%2u) = %3u,%3u,%3u,%3u  want %3u,%3u,%3u,%3u  %-19s %s\n",
               probes[i].region, probes[i].x, probes[i].y, px[0], px[1], px[2], px[3],
               probes[i].want[0], probes[i].want[1], probes[i].want[2], probes[i].want[3],
               probes[i].verdict, good ? "ok" : "MISMATCH");
    }
    for (int i = 0; i < 8; ++i)
        printf("  arm %d  %-20s expect %s\n", i + 1, arms[i].name, arms[i].expect);
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
