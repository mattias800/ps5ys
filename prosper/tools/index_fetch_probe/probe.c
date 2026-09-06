/* index_fetch_probe.c — bare-Vulkan reproducer for #2961 (prosper).
 *
 * Question it answers, with no prosper code and no vkprobe machinery anywhere
 * in the path: does THIS driver, on an indexed draw, hand the vertex shader
 * the fetched index-buffer value as gl_VertexIndex — or the sequential vertex
 * ordinal?
 *
 * Method: a POINTS draw whose vertex shader records the pair
 *   (ordinal = gl_VertexIndex, payload = vertex attribute fetched for that vertex)
 * and whose fragment shader stores results[payload] = ordinal. The vertex
 * buffer holds identity data (payload at source position p equals p), so on a
 * correct implementation every written slot reads back as its own index. A
 * slot whose value differs from its index — or a write landing on a slot that
 * was never indexed — is the divergence.
 *
 * Usage: index_fetch_probe            (runs the fixed issue-table lists)
 *
 * Exit code: 0 all lists fetched correctly, 1 divergence observed,
 * 2 setup/tool error. Positive control first: the identity list cannot
 * discriminate (both hypotheses agree on it) — it proves the harness draws.
 *
 * Build: linked by prosper's CMake as target `index_fetch_probe` (Vulkan only).
 */

#include <vulkan/vulkan.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SLOT_COUNT 256u          /* SSBO words */
#define SENTINEL   0xFFFFFFFFu

static const uint32_t kVertSpv[] = {
#include "probe_vert_spv.h"
};
static const uint32_t kFragSpv[] = {
#include "probe_frag_spv.h"
};

static VkInstance g_instance;
static VkPhysicalDevice g_pd;
static VkDevice g_dev;
static VkQueue g_queue;
static uint32_t g_queue_family;
static VkBuffer g_ssbo, g_vb, g_ib;
static VkDeviceMemory g_ssbo_mem, g_vb_mem, g_ib_mem;
static uint32_t *g_ssbo_map, *g_vb_map, *g_ib_map;
static VkDescriptorPool g_pool;
static VkDescriptorSet g_set;

static void die(const char *what, VkResult r)
{
    fprintf(stderr, "index_fetch_probe: %s failed (%d)\n", what, (int)r);
    exit(2);
}

static void require(int cond, const char *what)
{
    if (!cond) {
        fprintf(stderr, "index_fetch_probe: %s\n", what);
        exit(2);
    }
}

static uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t wanted,
                                 VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((wanted & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return UINT32_MAX;
}

static void make_buffer(VkBufferUsageFlags usage, VkDeviceSize size,
                        VkBuffer *buf, VkDeviceMemory *mem, uint32_t **map)
{
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r = vkCreateBuffer(g_dev, &bi, NULL, buf);
    if (r) die("vkCreateBuffer", r);

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, *buf, &mr);
    uint32_t type = find_memory_type(
        g_pd, mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    require(type != UINT32_MAX, "no host-visible memory type");

    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = type;
    r = vkAllocateMemory(g_dev, &ai, NULL, mem);
    if (r) die("vkAllocateMemory", r);
    r = vkBindBufferMemory(g_dev, *buf, *mem, 0);
    if (r) die("vkBindBufferMemory", r);
    r = vkMapMemory(g_dev, *mem, 0, size, 0, (void **)map);
    if (r) die("vkMapMemory", r);
}

static VkShaderModule make_shader(const uint32_t *words, size_t word_count)
{
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = word_count * sizeof(uint32_t);
    ci.pCode = words;
    VkShaderModule m;
    VkResult r = vkCreateShaderModule(g_dev, &ci, NULL, &m);
    if (r) die("vkCreateShaderModule", r);
    return m;
}

/* One probe list. Returns 1 if every unique index came back as itself, else 0. */
static int run_list(VkCommandBuffer cmd, VkPipeline pipeline,
                    VkPipelineLayout layout, VkRenderPass pass,
                    VkFramebuffer fb, const uint32_t *indices, uint32_t count,
                    char *report, size_t report_cap)
{
    /* clear results to sentinel, then fill the index buffer */
    for (uint32_t i = 0; i < SLOT_COUNT; ++i) g_ssbo_map[i] = SENTINEL;
    memcpy(g_ib_map, indices, count * sizeof(uint32_t));

    VkCommandBufferBeginInfo binfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &binfo);

    VkClearValue clear = {.color = {{0.f, 0.f, 0.f, 1.f}}};
    VkRenderPassBeginInfo rp = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = pass;
    rp.framebuffer = fb;
    rp.renderArea.extent.width = 64;
    rp.renderArea.extent.height = 64;
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0,
                            1, &g_set, 0, NULL);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &g_vb, &offset);
    vkCmdBindIndexBuffer(cmd, g_ib, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, count, 1, 0, 0, 0);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkResult r = vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE);
    if (r) die("vkQueueSubmit", r);
    vkQueueWaitIdle(g_queue);

    /* collect: slot -> stored ordinal */
    size_t off = 0;
    int ok = 1;
    uint32_t seen = 0;
    off += (size_t)snprintf(report + off, report_cap - off, "wrote {");
    for (uint32_t slot = 0; slot < SLOT_COUNT && off < report_cap; ++slot) {
        if (g_ssbo_map[slot] == SENTINEL) continue;
        ++seen;
        /* The slot must be one the index list actually named AND hold its own
         * value. Checking only self-consistency would pass ordinal-only
         * execution (an implementation ignoring the index buffer writes
         * {0:n, 1:n, ...} for identity data), which is exactly what this tool
         * exists to catch (#2961). */
        int indexed = 0;
        for (uint32_t j = 0; j < count; ++j)
            if (indices[j] == slot) { indexed = 1; break; }
        if (!indexed || g_ssbo_map[slot] != slot) ok = 0;
        off += (size_t)snprintf(report + off, report_cap - off, "%s%u:%u",
                                seen > 1 ? ", " : "", slot,
                                g_ssbo_map[slot]);
    }
    snprintf(report + off, report_cap - off, "} (%u writes)", seen);
    return ok && seen > 0;
}

#ifdef PROSPER_RENDERDOC_CONTROL
#include "renderdoc_control.h"
int main(int argc, char** argv)
#else
int main(void)
#endif
{
#ifdef PROSPER_RENDERDOC_CONTROL
    require(argc == 2, "usage: renderdoc_control <new capture path template>");
#endif
    setvbuf(stdout, NULL, _IONBF, 0);

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkResult r = vkCreateInstance(&ici, NULL, &g_instance);
    if (r) die("vkCreateInstance", r);

    uint32_t ndev = 0;
    vkEnumeratePhysicalDevices(g_instance, &ndev, NULL);
    require(ndev > 0, "no Vulkan physical device");
    VkPhysicalDevice devs[8];
    require(ndev <= 8, "too many devices");
    vkEnumeratePhysicalDevices(g_instance, &ndev, devs);
    g_pd = devs[0];

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL);
    require(nq > 0 && nq <= 16, "bad queue family count");
    VkQueueFamilyProperties qf[16];
    vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qf);
    for (uint32_t i = 0; i < nq; ++i)
        if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { g_queue_family = i; break; }

    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(g_pd, &features);
    require(features.fragmentStoresAndAtomics,
            "device lacks fragmentStoresAndAtomics (the FS stores to an SSBO)");

    float prio = 1.f;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = g_queue_family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    di.pEnabledFeatures = &features;
    r = vkCreateDevice(g_pd, &di, NULL, &g_dev);
    if (r) die("vkCreateDevice", r);
    vkGetDeviceQueue(g_dev, g_queue_family, 0, &g_queue);
#ifdef PROSPER_RENDERDOC_CONTROL
    capture_begin(argv[1]);
#endif

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_pd, &props);
    printf("== index_fetch_probe ==\ndevice: %s (driver %u.%u.%u)\n",
           props.deviceName, VK_VERSION_MAJOR(props.driverVersion),
           VK_VERSION_MINOR(props.driverVersion),
           VK_VERSION_PATCH(props.driverVersion));

    make_buffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                SLOT_COUNT * sizeof(uint32_t), &g_ssbo, &g_ssbo_mem,
                &g_ssbo_map);
    make_buffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                SLOT_COUNT * sizeof(uint32_t), &g_vb, &g_vb_mem, &g_vb_map);
    for (uint32_t i = 0; i < SLOT_COUNT; ++i) g_vb_map[i] = i; /* identity */
    make_buffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                1024 * sizeof(uint32_t), &g_ib, &g_ib_mem, &g_ib_map);

    VkDescriptorSetLayoutBinding binding = {0};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dli = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 1;
    dli.pBindings = &binding;
    VkDescriptorSetLayout set_layout;
    r = vkCreateDescriptorSetLayout(g_dev, &dli, NULL, &set_layout);
    if (r) die("vkCreateDescriptorSetLayout", r);

    VkPipelineLayoutCreateInfo pli = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &set_layout;
    VkPipelineLayout layout;
    r = vkCreatePipelineLayout(g_dev, &pli, NULL, &layout);
    if (r) die("vkCreatePipelineLayout", r);

    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pi = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 1;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    r = vkCreateDescriptorPool(g_dev, &pi, NULL, &g_pool);
    if (r) die("vkCreateDescriptorPool", r);
    VkDescriptorSetAllocateInfo dai = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g_pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &set_layout;
    r = vkAllocateDescriptorSets(g_dev, &dai, &g_set);
    if (r) die("vkAllocateDescriptorSets", r);

    VkDescriptorBufferInfo dbi = {g_ssbo, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = g_set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(g_dev, 1, &write, 0, NULL);

    VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;

    VkAttachmentDescription att = {0};
    att.format = color_format;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference cref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub = {0};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &cref;
    VkRenderPassCreateInfo rpi = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 1;
    rpi.pAttachments = &att;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &sub;
    VkRenderPass pass;
    r = vkCreateRenderPass(g_dev, &rpi, NULL, &pass);
    if (r) die("vkCreateRenderPass", r);

    VkShaderModule vs = make_shader(kVertSpv, sizeof(kVertSpv) / 4);
    VkShaderModule fs = make_shader(kFragSpv, sizeof(kFragSpv) / 4);
    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "vs_main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "fs_main";

    VkVertexInputBindingDescription vbind = {0, sizeof(uint32_t),
                                              VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription vattr = {0, 0, VK_FORMAT_R32_UINT, 0};
    VkPipelineVertexInputStateCreateInfo vi = {
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vbind;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions = &vattr;

    VkPipelineInputAssemblyStateCreateInfo ia = {
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkViewport vp = {0, 0, 64, 64, 0.f, 1.f};
    VkRect2D sc = {{0, 0}, {64, 64}};
    VkPipelineViewportStateCreateInfo vpstate = {
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vpstate.viewportCount = 1;
    vpstate.pViewports = &vp;
    vpstate.scissorCount = 1;
    vpstate.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rs = {
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth = 1.f;

    VkPipelineMultisampleStateCreateInfo ms = {
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba = {0};
    VkPipelineColorBlendStateCreateInfo cb = {
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkGraphicsPipelineCreateInfo gp = {
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vpstate;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb;
    gp.layout = layout;
    gp.renderPass = pass;
    VkPipeline pipeline;
    r = vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gp, NULL,
                                  &pipeline);
    if (r) die("vkCreateGraphicsPipelines", r);

    /* 64x64 offscreen color image + framebuffer */
    VkImageCreateInfo imi = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imi.imageType = VK_IMAGE_TYPE_2D;
    imi.format = color_format;
    imi.extent.width = 64;
    imi.extent.height = 64;
    imi.extent.depth = 1;
    imi.mipLevels = 1;
    imi.arrayLayers = 1;
    imi.samples = VK_SAMPLE_COUNT_1_BIT;
    imi.tiling = VK_IMAGE_TILING_OPTIMAL;
    imi.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VkImage image;
    r = vkCreateImage(g_dev, &imi, NULL, &image);
    if (r) die("vkCreateImage", r);
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_dev, image, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_memory_type(
        g_pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    require(ai.memoryTypeIndex != UINT32_MAX, "no device-local memory type");
    VkDeviceMemory image_mem;
    r = vkAllocateMemory(g_dev, &ai, NULL, &image_mem);
    if (r) die("vkAllocateMemory(image)", r);
    vkBindImageMemory(g_dev, image, image_mem, 0);

    VkImageViewCreateInfo vici = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vici.image = image;
    vici.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vici.format = color_format;
    vici.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vici.subresourceRange.levelCount = 1;
    vici.subresourceRange.layerCount = 1;
    VkImageView view;
    r = vkCreateImageView(g_dev, &vici, NULL, &view);
    if (r) die("vkCreateImageView", r);

    VkFramebufferCreateInfo fi = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fi.renderPass = pass;
    fi.attachmentCount = 1;
    fi.pAttachments = &view;
    fi.width = 64;
    fi.height = 64;
    fi.layers = 1;
    VkFramebuffer fb;
    r = vkCreateFramebuffer(g_dev, &fi, NULL, &fb);
    if (r) die("vkCreateFramebuffer", r);

    VkCommandPoolCreateInfo cpi = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = g_queue_family;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool;
    r = vkCreateCommandPool(g_dev, &cpi, NULL, &pool);
    if (r) die("vkCreateCommandPool", r);
    VkCommandBufferAllocateInfo cai = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    r = vkAllocateCommandBuffers(g_dev, &cai, &cmd);
    if (r) die("vkAllocateCommandBuffers", r);

    /* The issue's own table. [0,1,2] is the positive control: both hypotheses
     * agree there, so it cannot discriminate — it only proves we drew. */
    static const struct { const char *name; uint32_t idx[6]; uint32_t n; }
    lists[] = {
        {"control [0,1,2]", {0, 1, 2}, 3},
        {"[2,2,2]",         {2, 2, 2}, 3},
        {"[3,4,5]",         {3, 4, 5}, 3},
        {"[0,0,0]",         {0, 0, 0}, 3},
        {"[0,1,2,3,4,5]",   {0, 1, 2, 3, 4, 5}, 6},
    };

    int discriminating_ok = 1;
    for (size_t t = 0; t < sizeof(lists) / sizeof(lists[0]); ++t) {
        char report[512];
        int ok = run_list(cmd, pipeline, layout, pass, fb,
                          lists[t].idx, lists[t].n, report, sizeof(report));
        int is_control = (t == 0);
        printf("%-16s %s => %s\n", lists[t].name, report,
               ok ? "FETCHED (correct)"
                  : "ORDINAL (index buffer ignored)");
        if (!ok && !is_control) discriminating_ok = 0;
    }
    printf("verdict: %s\n",
           discriminating_ok ? "fetched correctly"
                             : "DIVERGENCE: indexed reads are ordinals");

    vkDeviceWaitIdle(g_dev);
#ifdef PROSPER_RENDERDOC_CONTROL
    capture_end();
#endif
    return discriminating_ok ? 0 : 1;
}
