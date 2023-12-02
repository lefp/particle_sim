#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cinttypes>
#include <cmath>

#include <sys/stat.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <loguru.hpp>

#include "types.hpp"
#include "error_util.hpp"
#include "vk_procs.hpp"
#include "defer.hpp"
#include "alloc_util.hpp"
#include "math_util.hpp"

//
// Global constants ==========================================================================================
//

#define VULKAN_API_VERSION VK_API_VERSION_1_3
#define SWAPCHAIN_FORMAT VK_FORMAT_B8G8R8A8_SRGB
#define SWAPCHAIN_COLOR_SPACE VK_COLOR_SPACE_SRGB_NONLINEAR_KHR

const char* APP_NAME = "an game";

// Use to represent an invalid queue family; can't use -1 (because unsigned) or 0 (because it's valid).
const u32 INVALID_QUEUE_FAMILY_IDX = UINT32_MAX;
const u32 INVALID_PHYSICAL_DEVICE_IDX = UINT32_MAX;

// Use to statically allocate arrays for swapchain images, image views, and framebuffers.
const u32 MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT = 4; // just picked a probably-reasonable number, idk

const VkExtent2D DEFAULT_WINDOW_EXTENT { 800, 600 };

//
// Global variables ==========================================================================================
//

static VkInstance instance_ = VK_NULL_HANDLE;
static VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
static VkPhysicalDeviceProperties physical_device_properties_;
static u32 queue_family_ = INVALID_QUEUE_FAMILY_IDX;
static VkDevice device_ = VK_NULL_HANDLE;
static VkQueue queue_ = VK_NULL_HANDLE;

static struct {
    VkPipeline voxel_pipeline = VK_NULL_HANDLE;
} pipelines_;

static struct {
    VkPipelineLayout voxel_pipeline_layout = VK_NULL_HANDLE;
} pipeline_layouts_;

static VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_FIFO_KHR;

//
// ===========================================================================================================
//

struct QueueFamilyRequirements {
    VkQueueFlags required_queue_flags;
    bool require_presentation_support;
};

struct PhysicalDeviceTypePriorities {
    u8 other;
    u8 integrated_gpu;
    u8 discrete_gpu;
    u8 virtual_gpu;
    u8 cpu;

    u8 getPriority(VkPhysicalDeviceType type) const {
        // @note This implementation relies on the PhysicalDeviceType enum values specified in VK spec v1.3.234.
        // We use the PhysicalDeviceType enum as an index.

        alwaysAssert(0 <= type && type <= 4);
        const u8* priorities = (u8*)this;
        return priorities[type];
    }
};

struct VoxelPipelineVertexShaderPushConstants {
    f32 angle_radians;
};

//
// ===========================================================================================================
//

void _assertGlfw(bool condition, const char* file, int line) {

    if (condition) return;


    const char* err_description = NULL;
    int err_code = glfwGetError(&err_description);
    if (err_description == NULL) err_description = "(NO DESCRIPTION PROVIDED)";

    ABORT_F(
        "Assertion failed! GLFW error code %i, file `%s`, line %i, description `%s`",
        err_code, file, line, err_description
    );
};
#define assertGlfw(condition) _assertGlfw(condition, __FILE__, __LINE__)


void _abortIfGlfwError(const char* file, int line) {

    const char* err_description = NULL;
    int err_id = glfwGetError(&err_description);
    if (err_id == GLFW_NO_ERROR) return;

    if (err_description == NULL) err_description = "(NO DESCRIPTION PROVIDED)";
    ABORT_F(
        "Aborting due to GLFW error code %i, file `%s`, line %i, description `%s`",
        err_id, file, line, err_description
    );
};
#define abortIfGlfwError() _abortIfGlfwError(__FILE__, __LINE__)


void _assertVk(VkResult result, const char* file, int line) {

    if (result == VK_SUCCESS) return;

    LOG_F(
        FATAL, "VkResult is %i, file `%s`, line %i",
        result, file, line
    );
    abort();
}
#define assertVk(result) _assertVk(result, __FILE__, __LINE__)


bool flagsSubset(VkQueueFlags subset, VkQueueFlags superset) {
    return (subset & superset) == subset;
}


/// If no satisfactory family found, returns `QUEUE_FAMILY_NOT_FOUND`.
u32 firstSatisfactoryQueueFamily(
      VkInstance instance,
      VkPhysicalDevice device,
      u32 family_count,
      const VkQueueFamilyProperties* family_properties_list,
      const QueueFamilyRequirements* requirements
) {
    for (u32fast fam_idx = 0; fam_idx < family_count; fam_idx++) {

        VkQueueFamilyProperties fam_props = family_properties_list[fam_idx];
        if (!flagsSubset(requirements->required_queue_flags, fam_props.queueFlags)) continue;

        if (requirements->require_presentation_support) {
            bool supports_present = glfwGetPhysicalDevicePresentationSupport(instance, device, fam_idx);
            if (!supports_present) {
                abortIfGlfwError(); // check if present support returned false due to an error
                continue;
            }
        };

        return fam_idx;
    }

    return INVALID_QUEUE_FAMILY_IDX;
}


/// If no satisfactory device is found, `device_out` is set to `VK_NULL_HANDLE`.
/// You may request a specific physical device using `specific_physical_device_request`.
///     That device is selected iff it exists and satisfies requirements (ignoring `device_type_priorities`).
///     To avoid requesting a specific device, pass `INVALID_PHYSICAL_DEVICE_IDX`.
/// `device_type_priorities` are interpreted as follows:
///     0 means "do not use".
///     A higher number indicates greater priority.
/// Returns the index of the selected device.
void selectPhysicalDeviceAndQueueFamily(
    VkInstance instance,
    u32* device_idx_out,
    u32* queue_family_out,
    u32 device_count,
    const VkPhysicalDevice* devices,
    const VkPhysicalDeviceProperties* device_properties_list,
    const QueueFamilyRequirements* queue_family_requirements,
    PhysicalDeviceTypePriorities device_type_priorities,
    u32 specific_device_request
) {
    u32 current_best_device_idx = INVALID_PHYSICAL_DEVICE_IDX;
    u8 current_best_device_priority = 0;
    u32 current_best_device_queue_family = INVALID_QUEUE_FAMILY_IDX;

    for (u32 dev_idx = 0; dev_idx < device_count; dev_idx++) {

        const VkPhysicalDevice device = devices[dev_idx];


        const VkPhysicalDeviceProperties device_props = device_properties_list[dev_idx];

        const u8 device_priority = device_type_priorities.getPriority(device_props.deviceType);
        if (device_priority <= current_best_device_priority and dev_idx != specific_device_request) continue;


        u32 family_count = 0;
        vk_inst_procs.getPhysicalDeviceQueueFamilyProperties(device, &family_count, NULL);
        alwaysAssert(family_count > 0);

        VkQueueFamilyProperties* family_props_list = mallocArray<VkQueueFamilyProperties>(family_count);
        defer(free(family_props_list));
        vk_inst_procs.getPhysicalDeviceQueueFamilyProperties(device, &family_count, family_props_list);

        const u32 fam = firstSatisfactoryQueueFamily(
            instance, device, family_count, family_props_list, queue_family_requirements
        );
        if (fam == INVALID_QUEUE_FAMILY_IDX) {
            LOG_F(INFO, "Physical device %" PRIu32 "has no satisfactory queue family.", dev_idx);
            continue;
        }


        current_best_device_idx = dev_idx;
        current_best_device_priority = device_priority;
        current_best_device_queue_family = fam;

        if (dev_idx == specific_device_request) break;
    }

    *device_idx_out = current_best_device_idx;
    *queue_family_out = current_best_device_queue_family;
}


/// If `specific_device_request` isn't NULL, attempts to select a device with that name.
/// If no such device exists or doesn't satisfactory requirements, silently selects a different device.
void initGraphicsUptoQueueCreation(const char* specific_named_device_request) {

    if (!glfwVulkanSupported()) ABORT_F("Failed to find Vulkan; do you need to install drivers?");
    auto vkCreateInstance = (PFN_vkCreateInstance)glfwGetInstanceProcAddress(NULL, "vkCreateInstance");
    assertGlfw(vkCreateInstance != NULL);

    // Create instance ---------------------------------------------------------------------------------------
    {
        u32 extensions_required_by_glfw_count = 0;
        const char** extensions_required_by_glfw = glfwGetRequiredInstanceExtensions(&extensions_required_by_glfw_count);
        assertGlfw(extensions_required_by_glfw != NULL);

        #ifndef NDEBUG
            const u32 instance_layer_count = 1;
            const char* instance_layers[instance_layer_count] = {"VK_LAYER_KHRONOS_validation"};
        #else
            const u32 instance_layer_count = 0;
            const char* instance_layers[instance_layer_count] = {};
        #endif

        VkApplicationInfo app_info {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = APP_NAME,
            .apiVersion = VULKAN_API_VERSION,
        };

        VkInstanceCreateInfo instance_info {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = instance_layer_count,
            .ppEnabledLayerNames = instance_layers,
            .enabledExtensionCount = extensions_required_by_glfw_count,
            .ppEnabledExtensionNames = extensions_required_by_glfw,
        };

        VkResult result = vkCreateInstance(&instance_info, NULL, &instance_);
        assertVk(result);

        vk_inst_procs.init(instance_, (PFN_vkGetInstanceProcAddr)glfwGetInstanceProcAddress);
    }

    // Select physical device and queue families -------------------------------------------------------------
    {
        u32 physical_device_count = 0;
        VkResult result = vk_inst_procs.enumeratePhysicalDevices(instance_, &physical_device_count, NULL);
        assertVk(result);

        if (physical_device_count == 0) ABORT_F("Found no Vulkan devices.");


        VkPhysicalDevice* physical_devices = mallocArray<VkPhysicalDevice>(physical_device_count);
        defer(free(physical_devices));

        result = vk_inst_procs.enumeratePhysicalDevices(instance_, &physical_device_count, physical_devices);
        assertVk(result);


        VkPhysicalDeviceProperties* physical_device_properties_list =
            mallocArray<VkPhysicalDeviceProperties>(physical_device_count);
        defer(free(physical_device_properties_list));

        u32 requested_device_idx = INVALID_PHYSICAL_DEVICE_IDX;
        for (u32 dev_idx = 0; dev_idx < physical_device_count; dev_idx++) {

            VkPhysicalDeviceProperties* p_dev_props = &physical_device_properties_list[dev_idx];
            vk_inst_procs.getPhysicalDeviceProperties(physical_devices[dev_idx], p_dev_props);

            const char* device_name = p_dev_props->deviceName;
            LOG_F(INFO, "Found physical device %u: `%s`.", dev_idx, device_name);
            if (
                specific_named_device_request != NULL and
                strcmp(specific_named_device_request, device_name) == 0
            ) {
                LOG_F(INFO, "Physical device %u: name matches requested device.", dev_idx);
                requested_device_idx = dev_idx;
            }
        }
        LOG_IF_F(
            WARNING,
            specific_named_device_request != NULL and requested_device_idx == INVALID_PHYSICAL_DEVICE_IDX,
            "Requested device named `%s` not found.", specific_named_device_request
        );


        // NOTE: The Vulkan spec doesn't guarantee that there is a single queue family that supports both
        // Graphics and Present. But in practice, I expect every device that supports Present to have a family
        // supporting both.
        QueueFamilyRequirements queue_family_requirements {
            .required_queue_flags = VK_QUEUE_GRAPHICS_BIT, // TODO add VK_QUEUE_COMPUTE_BIT, once it's needed
            .require_presentation_support = true,
        };

        PhysicalDeviceTypePriorities device_type_priorities {
            .other = 0,
            .integrated_gpu = 1,
            .discrete_gpu = 2,
            .virtual_gpu = 0,
            .cpu = 0,
        };

        u32 physical_device_idx = INVALID_PHYSICAL_DEVICE_IDX;
        selectPhysicalDeviceAndQueueFamily(
            instance_,
            &physical_device_idx, &queue_family_,
            physical_device_count, physical_devices, physical_device_properties_list,
            &queue_family_requirements, device_type_priorities,
            requested_device_idx
        );
        alwaysAssert(physical_device_idx != INVALID_PHYSICAL_DEVICE_IDX);
        physical_device_ = physical_devices[physical_device_idx];

        vk_inst_procs.getPhysicalDeviceProperties(physical_device_, &physical_device_properties_);
        LOG_F(INFO, "Selected physical device `%s`.", physical_device_properties_.deviceName);
        LOG_IF_F(
            WARNING,
            specific_named_device_request != NULL and physical_device_idx != requested_device_idx,
            "Didn't select requested device named `%s`.", specific_named_device_request
        );
    }

    // Create logical device and queues ----------------------------------------------------------------------
    {
        // NOTE: using only 1 queue, because some cards only provide 1 queue.
        // (E.g. the Intel integrated graphics on my laptop.)
        const u32fast queue_count = 1;
        const f32 queue_priorities[queue_count] = { 1.0 };
        VkDeviceQueueCreateInfo queue_cinfo {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queue_family_,
            .queueCount = queue_count,
            .pQueuePriorities = queue_priorities,
        };

        const u32 device_extension_count = 1;
        const char* device_extensions[] = { "VK_KHR_swapchain" };

        VkDeviceCreateInfo device_cinfo {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_cinfo,
            .enabledExtensionCount = device_extension_count,
            .ppEnabledExtensionNames = device_extensions,
        };

        VkResult result = vk_inst_procs.createDevice(physical_device_, &device_cinfo, NULL, &device_);
        assertVk(result);

        vk_dev_procs.init(device_, vk_inst_procs.getDeviceProcAddr);

        // NOTE: Vk Spec 1.3.259:
        //     vkGetDeviceQueue must only be used to get queues that were created with the `flags` parameter
        //     of VkDeviceQueueCreateInfo set to zero.
        vk_dev_procs.getDeviceQueue(device_, queue_family_, 0, &queue_);
    }
}


/// You own the returned buffer. You may free it using `free()`.
/// On error, either aborts or returns `NULL`.
void* readEntireFile(const char* fname, size_t* size_out) {
    // TODO: Maybe using `open()`, `fstat()`, and `read()` would be faster; because we don't need buffered
    // input, and maybe using `fseek()` to get the file size is unnecessarily slow.

    FILE* file = fopen(fname, "r");
    if (file == NULL) {
        LOG_F(ERROR, "Failed to open file `%s`; errno: `%i`, description: `%s`.", fname, errno, strerror(errno));
        return NULL;
    }

    int result = fseek(file, 0, SEEK_END);
    assertErrno(result == 0);

    long file_size = ftell(file);
    assertErrno(file_size >= 0);

    result = fseek(file, 0, SEEK_SET);
    assertErrno(result == 0);


    void* buffer = malloc(file_size);
    assertErrno(buffer != NULL);

    size_t n_items_read = fread(buffer, file_size, 1, file);
    alwaysAssert(n_items_read == 1);

    result = fclose(file);
    assertErrno(result == 0);


    *size_out = file_size;
    return buffer;
}


VkShaderModule createShaderModuleFromSpirvFile(const char* spirv_fname, VkDevice device) {

    size_t spirv_size_bytes = 0;
    void* spirv_buffer = readEntireFile(spirv_fname, &spirv_size_bytes);

    if (spirv_buffer == NULL) return VK_NULL_HANDLE;
    defer(free(spirv_buffer));

    alwaysAssert(spirv_size_bytes % 4 == 0); // spirv is a stream of `u32`s.


    VkShaderModuleCreateInfo cinfo {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_size_bytes,
        .pCode = (u32*)spirv_buffer,
    };

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkResult result = vk_dev_procs.createShaderModule(device, &cinfo, NULL, &shader_module);
    assertVk(result);

    return shader_module;
};


VkRenderPass createSimpleRenderPass(VkDevice device) {
    constexpr u32 attachment_count = 1;
    const VkAttachmentDescription attachment_descriptions[attachment_count] {
        {
            .format = SWAPCHAIN_FORMAT,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        }
    };

    constexpr u32 color_attachment_count = 1;
    const VkAttachmentReference color_attachments[color_attachment_count] {
        {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        }
    };

    constexpr u32 subpass_count = 1;
    const VkSubpassDescription subpass_descriptions[subpass_count] {
        {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .pInputAttachments = NULL,
            .colorAttachmentCount = color_attachment_count,
            .pColorAttachments = color_attachments,
            .pResolveAttachments = NULL,
            .pDepthStencilAttachment = NULL,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = NULL,
        }
    };

    const VkRenderPassCreateInfo render_pass_info {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = attachment_count,
        .pAttachments = attachment_descriptions,
        .subpassCount = subpass_count,
        .pSubpasses = subpass_descriptions,
        .dependencyCount = 0,
        .pDependencies = NULL,
    };

    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkResult result = vk_dev_procs.createRenderPass(device, &render_pass_info, NULL, &render_pass);
    assertVk(result);

    return render_pass;
}


VkPipeline createVoxelPipeline(
    VkDevice device,
    VkRenderPass render_pass,
    u32 subpass,
    VkPipelineLayout* pipeline_layout_out
) {

    VkShaderModule vertex_shader_module = createShaderModuleFromSpirvFile("build/voxel.vert.spv", device);
    alwaysAssert(vertex_shader_module != VK_NULL_HANDLE);
    defer(vk_dev_procs.destroyShaderModule(device, vertex_shader_module, NULL));

    VkShaderModule fragment_shader_module = createShaderModuleFromSpirvFile("build/voxel.frag.spv", device);
    alwaysAssert(fragment_shader_module != VK_NULL_HANDLE);
    defer(vk_dev_procs.destroyShaderModule(device, fragment_shader_module, NULL));

    constexpr u32 shader_stage_info_count = 2;
    const VkPipelineShaderStageCreateInfo shader_stage_infos[shader_stage_info_count] {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module,
            .pName = "main",
        },
    };


    const VkPipelineVertexInputStateCreateInfo vertex_input_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = NULL,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = NULL,
    };


    const VkPipelineInputAssemblyStateCreateInfo input_assembly_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };


    const VkPipelineViewportStateCreateInfo viewport_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = NULL, // using dynamic viewport
        .scissorCount = 1,
        .pScissors = NULL, // using dynamic scissor
    };


    const VkPipelineRasterizationStateCreateInfo rasterization_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0,
        .depthBiasClamp = 0.0,
        .depthBiasSlopeFactor = 0.0,
        .lineWidth = 1.0,
    };


    const VkPipelineMultisampleStateCreateInfo multisample_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 0.0,
        .pSampleMask = NULL,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };


    const VkPipelineDepthStencilStateCreateInfo depth_stencil_state_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_FALSE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_NEVER,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = {},
        .back = {},
        .minDepthBounds = 0.0,
        .maxDepthBounds = 1.0,
    };


    const VkPipelineColorBlendAttachmentState color_blend_attachment_info {
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blend_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_CLEAR,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment_info,
        .blendConstants = {0.0, 0.0, 0.0, 0.0},
    };


    constexpr u32 dynamic_state_count = 2;
    const VkDynamicState dynamic_states[dynamic_state_count] {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    const VkPipelineDynamicStateCreateInfo dynamic_state_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = dynamic_state_count,
        .pDynamicStates = dynamic_states,
    };


    constexpr u32 push_constant_range_count = 1;
    VkPushConstantRange push_constant_ranges[push_constant_range_count] {
        VkPushConstantRange {
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(VoxelPipelineVertexShaderPushConstants),
        },
    };

    const VkPipelineLayoutCreateInfo pipeline_layout_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 0,
        .pSetLayouts = NULL,
        .pushConstantRangeCount = push_constant_range_count,
        .pPushConstantRanges = push_constant_ranges,
    };

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkResult result = vk_dev_procs.createPipelineLayout(device, &pipeline_layout_info, NULL, &pipeline_layout);
    assertVk(result);
    *pipeline_layout_out = pipeline_layout;


    const VkGraphicsPipelineCreateInfo pipeline_info {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = shader_stage_info_count,
        .pStages = shader_stage_infos,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly_info,
        .pTessellationState = NULL,
        .pViewportState = &viewport_info,
        .pRasterizationState = &rasterization_info,
        .pMultisampleState = &multisample_info,
        .pDepthStencilState = &depth_stencil_state_info,
        .pColorBlendState = &color_blend_info,
        .pDynamicState = &dynamic_state_info,
        .layout = pipeline_layout,
        .renderPass = render_pass,
        .subpass = subpass,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    VkPipeline graphics_pipeline = VK_NULL_HANDLE;
    result = vk_dev_procs.createGraphicsPipelines(
        device,
        VK_NULL_HANDLE, // pipelineCache
        1, // createInfoCount
        &pipeline_info,
        NULL, // allocationCallbacks
        &graphics_pipeline
    );
    assertVk(result);

    return graphics_pipeline;
}


/// Doesn't verify surface support. You must do so before calling this.
/// `fallback_extent` is used if the surface doesn't report a specific extent via Vulkan surface properties.
///     If you want resizing to work on all platforms, you should probably get the window's current dimensions
///     from your window provider. Some providers supposedly don't report their current dimensions via Vulkan,
///     according to
///     https://gist.github.com/nanokatze/bb03a486571e13a7b6a8709368bd87cf#file-handling-window-resize-md
/// Doesn't destroy `old_swapchain`; simply retires it. You are responsible for destroying it.
/// `old_swapchain` may be `VK_NULL_HANDLE`.
/// `extent_out` must not be NULL.
VkSwapchainKHR createSwapchain(
     VkPhysicalDevice physical_device,
     VkDevice device,
     VkSurfaceKHR surface,
     VkExtent2D fallback_extent,
     u32 queue_family_index,
     VkPresentModeKHR present_mode,
     VkSwapchainKHR old_swapchain,
     VkExtent2D* extent_out
) {

    VkSurfaceCapabilitiesKHR surface_capabilities {};
    VkResult result = vk_inst_procs.getPhysicalDeviceSurfaceCapabilitiesKHR(
        physical_device, surface, &surface_capabilities
    );
    assertVk(result);


    // Vk spec 1.3.259, appendix VK_KHR_swapchain, issue 12: suggests using capabilities.minImageCount + 1
    // to guarantee that vkAcquireNextImageKHR is non-blocking when using Mailbox present mode.
    // Idk what effect this has on FIFO present mode, but I'm assuming it doesn't hurt.
    u32 min_image_count = surface_capabilities.minImageCount + 1;
    {
        u32 count_preclamp = min_image_count;
        min_image_count = math::max(min_image_count, surface_capabilities.minImageCount);
        if (surface_capabilities.maxImageCount != 0) {
            min_image_count = math::min(min_image_count, surface_capabilities.maxImageCount);
        }

        if (min_image_count != count_preclamp) LOG_F(
            WARNING, "Min swapchain image count clamped from %u to %u, to fit surface limits.",
            count_preclamp, min_image_count
        );
    }


    // Vk spec 1.3.234:
    //     On some platforms, it is normal that maxImageExtent may become (0, 0), for example when the window
    //     is minimized. In such a case, it is not possible to create a swapchain due to the Valid Usage
    //     requirements.
    const VkExtent2D max_extent = surface_capabilities.maxImageExtent;
    if (max_extent.height == 0 and max_extent.width == 0) {
        ABORT_F("Surface maxImageExtent reported as (0, 0)."); // TODO handle this properly instead of aborting
    }

    VkExtent2D extent = surface_capabilities.currentExtent;
    // Vk spec 1.3.234:
    //     currentExtent is the current width and height of the surface, or the special value (0xFFFFFFFF,
    //     0xFFFFFFFF) indicating that the surface size will be determined by the extent of a swapchain
    //     targeting the surface.
    if (extent.width == 0xFF'FF'FF'FF and extent.height == 0xFF'FF'FF'FF) {

        LOG_F(INFO, "Surface currentExtent is (0xFFFFFFFF, 0xFFFFFFFF); using fallback extent.");

        const VkExtent2D min_extent = surface_capabilities.minImageExtent;
        const VkExtent2D max_extent = surface_capabilities.maxImageExtent;

        extent.width = math::clamp(fallback_extent.width, min_extent.width, max_extent.width);
        extent.height = math::clamp(fallback_extent.height, min_extent.height, max_extent.height);
        if (extent.width != fallback_extent.width or extent.height != fallback_extent.height) LOG_F(
            WARNING, "Adjusted fallback swapchain extent (%u, %u) to (%u, %u), to fit surface limits.",
            fallback_extent.width, fallback_extent.height, extent.width, extent.height
        );
    }


    // Vk spec 1.3.259 guarantees that this is true, but just in case.
    alwaysAssert(surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);


    VkSwapchainCreateInfoKHR swapchain_info {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = min_image_count,
        .imageFormat = SWAPCHAIN_FORMAT,
        .imageColorSpace = SWAPCHAIN_COLOR_SPACE,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices = &queue_family_index,
        .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode,
        .clipped = VK_FALSE,
        .oldSwapchain = old_swapchain,
    };

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    result = vk_dev_procs.createSwapchainKHR(device, &swapchain_info, NULL, &swapchain);
    assertVk(result);

    LOG_F(INFO, "Built swapchain.");
    *extent_out = extent;
    return swapchain;
}


/// Writes at most `max_image_count` to `images_out`.
/// Returns the actual number of images in the swapchain.
u32 getSwapchainImages(
    VkDevice device,
    VkSwapchainKHR swapchain,
    u32 max_image_count,
    VkImage* images_out
) {

    u32 swapchain_image_count = 0;
    VkResult result = vk_dev_procs.getSwapchainImagesKHR(device, swapchain, &swapchain_image_count, NULL);
    assertVk(result);

    bool max_too_small = max_image_count < swapchain_image_count;

    result = vk_dev_procs.getSwapchainImagesKHR(device, swapchain, &max_image_count, images_out);
    if (max_too_small and result != VK_INCOMPLETE)
        ABORT_F("Expected VkResult VK_INCOMPLETE (%i), got %i.", VK_INCOMPLETE, result);
    else assertVk(result);

    return swapchain_image_count;
}


/// Returns whether it succeeded.
bool createImageViewsForSwapchain(
    VkDevice device,
    u32fast image_count,
    const VkImage* swapchain_images,
    VkImageView* image_views_out
) {

    for (u32fast im_idx = 0; im_idx < image_count; im_idx++) {

        const VkImage image = swapchain_images[im_idx];
        VkImageView* p_image_view = &image_views_out[im_idx];

        const VkComponentMapping component_mapping {
            .r = VK_COMPONENT_SWIZZLE_R,
            .g = VK_COMPONENT_SWIZZLE_G,
            .b = VK_COMPONENT_SWIZZLE_B,
            .a = VK_COMPONENT_SWIZZLE_A
        };

        const VkImageSubresourceRange subresource_range = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        const VkImageViewCreateInfo image_view_info {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = SWAPCHAIN_FORMAT,
            .components = component_mapping,
            .subresourceRange = subresource_range,
        };

        VkResult result = vk_dev_procs.createImageView(device, &image_view_info, NULL, p_image_view);
        assertVk(result);
    }

    return true;
}


/// Returns whether it succeeded.
bool createFramebuffersForSwapchain(
    VkDevice device,
    VkRenderPass render_pass,
    VkExtent2D swapchain_extent,
    u32fast image_view_count,
    const VkImageView* image_views,
    VkFramebuffer* framebuffers_out
) {

    for (u32fast fb_idx = 0; fb_idx < image_view_count; fb_idx++) {

        const VkImageView* p_image_view = &image_views[fb_idx];
        VkFramebuffer* p_framebuffer = &framebuffers_out[fb_idx];

        const VkFramebufferCreateInfo framebuffer_info {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = p_image_view,
            .width = swapchain_extent.width,
            .height = swapchain_extent.height,
            .layers = 1,
        };

        VkResult result = vk_dev_procs.createFramebuffer(device, &framebuffer_info, NULL, p_framebuffer);
        assertVk(result);
    }

    return true;
}


/// Returns a 16:9 subregion centered in an image, which maximizes the subregion's area.
VkRect2D centeredSubregion_16x9(VkExtent2D image_extent) {

    const bool limiting_dim_is_x = image_extent.width * 9 <= image_extent.height * 16;

    VkOffset2D offset {};
    VkExtent2D extent {};
    if (limiting_dim_is_x) {
        extent.width = image_extent.width;
        extent.height = image_extent.width * 9 / 16;
        offset.x = 0;
        offset.y = (image_extent.height - extent.height) / 2;
    }
    else {
        extent.height = image_extent.height;
        extent.width = image_extent.height * 16 / 9;
        offset.y = 0;
        offset.x = (image_extent.width - extent.width) / 2;
    }

    return VkRect2D {
        .offset = offset,
        .extent = extent,
    };
}


/// Returns `true` if successful.
bool recordVoxelCommandBuffer(
    VkCommandBuffer command_buffer,
    VkRenderPass render_pass,
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    VkExtent2D swapchain_extent,
    VkRect2D swapchain_roi,
    VkFramebuffer framebuffer,
    const VoxelPipelineVertexShaderPushConstants* push_constants
) {

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VkResult result = vk_dev_procs.beginCommandBuffer(command_buffer, &begin_info);
    assertVk(result);

    {
        VkClearValue clear_value { .color = VkClearColorValue { .float32 = {0, 0, 0, 1} } };
        VkRenderPassBeginInfo render_pass_begin_info {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render_pass,
            .framebuffer = framebuffer,
            .renderArea = VkRect2D { .offset = {0, 0}, .extent = swapchain_extent },
            .clearValueCount = 1,
            .pClearValues = &clear_value,
        };
        vk_dev_procs.cmdBeginRenderPass(
            command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE
        );

        vk_dev_procs.cmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        const VkViewport viewport {
            .x = (f32)swapchain_roi.offset.x,
            .y = (f32)swapchain_roi.offset.y,
            .width = (f32)swapchain_roi.extent.width,
            .height = (f32)swapchain_roi.extent.height,
            .minDepth = 0,
            .maxDepth = 1,
        };
        vk_dev_procs.cmdSetViewport(command_buffer, 0, 1, &viewport);
        vk_dev_procs.cmdSetScissor(command_buffer, 0, 1, &swapchain_roi);

        vk_dev_procs.cmdPushConstants(
            command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
            sizeof(*push_constants), push_constants
        );

        vk_dev_procs.cmdDraw(command_buffer, 36, 1, 0, 0);

        vk_dev_procs.cmdEndRenderPass(command_buffer);
    }

    result = vk_dev_procs.endCommandBuffer(command_buffer);
    assertVk(result);

    return true;
}


int main(int argc, char** argv) {

    loguru::init(argc, argv);
    #ifndef NDEBUG
        LOG_F(INFO, "Debug build.");
    #else
        LOG_F(INFO), "Release build.");
    #endif

    int success = glfwInit();
    assertGlfw(success);


    const char* specific_device_request = getenv("PHYSICAL_DEVICE_NAME");
    initGraphicsUptoQueueCreation(specific_device_request);

    VkRenderPass render_pass = createSimpleRenderPass(device_);
    alwaysAssert(render_pass != VK_NULL_HANDLE);

    const u32 subpass = 0;
    VkPipeline pipeline = createVoxelPipeline(
        device_, render_pass, subpass, &pipeline_layouts_.voxel_pipeline_layout
    );
    alwaysAssert(pipeline != VK_NULL_HANDLE);
    pipelines_.voxel_pipeline = pipeline;

    // TODO remaining work:
    // Set up validation layer debug logging thing, to log their messages as loguru messages. This way they
    // will be logged if we're logging to a file.


    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // don't initialize OpenGL, because we're using Vulkan
    GLFWwindow* window = glfwCreateWindow(
        DEFAULT_WINDOW_EXTENT.width, DEFAULT_WINDOW_EXTENT.height, "an game", NULL, NULL
    );
    assertGlfw(window != NULL);

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult result = glfwCreateWindowSurface(instance_, window, NULL, &surface);
    assertVk(result);

    int current_window_width = 0;
    int current_window_height = 0;
    glfwGetWindowSize(window, &current_window_width, &current_window_height);
    abortIfGlfwError();
    // TODO if current_width == current_height == 0, check if window is minimized or something; if it is, do
    // something that doesn't waste resources

    VkExtent2D swapchain_extent {};
    VkSwapchainKHR swapchain = createSwapchain(
        physical_device_,
        device_,
        surface,
        VkExtent2D { .width = (u32)current_window_width, .height = (u32)current_window_height },
        queue_family_,
        present_mode_,
        VK_NULL_HANDLE, // old_swapchain
        &swapchain_extent
    );
    alwaysAssert(swapchain != VK_NULL_HANDLE);

    VkRect2D swapchain_roi = centeredSubregion_16x9(swapchain_extent);


    VkImage p_swapchain_images[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkImageView p_swapchain_image_views[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkFramebuffer p_simple_render_pass_swapchain_framebuffers[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer p_command_buffers[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkFence p_command_buffer_pending_fences[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkSemaphore p_render_finished_semaphores[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    VkSemaphore p_swapchain_image_acquired_semaphores[MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT] {};
    u32 swapchain_image_count = 0;
    {
        swapchain_image_count = getSwapchainImages(
            device_, swapchain, MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT, p_swapchain_images
        );
        if (swapchain_image_count > MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT) ABORT_F(
            "Unexpectedly large swapchain image count; assumed at most %u, actually %u.",
            MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT, swapchain_image_count
        );

        success = createImageViewsForSwapchain(
            device_, swapchain_image_count, p_swapchain_images, p_swapchain_image_views
        );
        alwaysAssert(success);

        success = createFramebuffersForSwapchain(
            device_, render_pass, swapchain_extent, swapchain_image_count, p_swapchain_image_views,
            p_simple_render_pass_swapchain_framebuffers
        );
        alwaysAssert(success);


        const VkCommandPoolCreateInfo command_pool_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family_,
        };

        result = vk_dev_procs.createCommandPool(device_, &command_pool_info, NULL, &command_pool);
        assertVk(result);

        VkCommandBufferAllocateInfo command_buffer_alloc_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = swapchain_image_count,
        };

        result = vk_dev_procs.allocateCommandBuffers(device_, &command_buffer_alloc_info, p_command_buffers);
        assertVk(result);


        VkFenceCreateInfo fence_info {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        for (u32 im_idx = 0; im_idx < swapchain_image_count; im_idx++) {
            result = vk_dev_procs.createFence(
                device_, &fence_info, NULL, &p_command_buffer_pending_fences[im_idx]
            );
            assertVk(result);
        }

        VkSemaphoreCreateInfo semaphore_info { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        for (u32 im_idx = 0; im_idx < swapchain_image_count; im_idx++) {
            result = vk_dev_procs.createSemaphore(
                device_, &semaphore_info, NULL, &p_render_finished_semaphores[im_idx]
            );
            assertVk(result);
        }
        for (u32 im_idx = 0; im_idx < swapchain_image_count; im_idx++) {
            result = vk_dev_procs.createSemaphore(
                device_, &semaphore_info, NULL, &p_swapchain_image_acquired_semaphores[im_idx]
            );
            assertVk(result);
        }
    }


    bool swapchain_needs_rebuild = false;
    u32fast frame_counter = 0;

    glfwPollEvents();
    while (!glfwWindowShouldClose(window)) {

        // TODO Make sure this makes sense, and if so, write down why it makes sense here.
        VkSemaphore swapchain_image_acquired_semaphore =
            p_swapchain_image_acquired_semaphores[frame_counter % swapchain_image_count];

        // Acquire swapchain image; if out of date, rebuild swapchain until it works. ------------------------

        u32 swapchain_rebuild_count = 0;
        u32 acquired_swapchain_image_idx = UINT32_MAX;
        while (true) {

            if (swapchain_needs_rebuild) {

                int window_width = 0;
                int window_height = 0;
                glfwGetWindowSize(window, &window_width, &window_height);
                abortIfGlfwError();
                // TODO if current_width == current_height == 0, check if window is minimized or something;
                // if it is, do something that doesn't waste resources

                VkSwapchainKHR old_swapchain = swapchain;
                swapchain = createSwapchain(
                    physical_device_, device_, surface,
                    VkExtent2D { .width = (u32)window_width, .height = (u32)window_height },
                    queue_family_, present_mode_, swapchain, &swapchain_extent
                );
                alwaysAssert(swapchain != VK_NULL_HANDLE);

                swapchain_rebuild_count++;

                // Before destroying the old swapchain, make sure it's not in use.
                result = vk_dev_procs.queueWaitIdle(queue_);
                assertVk(result);
                vk_dev_procs.destroySwapchainKHR(device_, old_swapchain, NULL);
            }

            // TODO: (2023-11-20) The only purpose of this block is to work around this validation layer bug:
            // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/6068
            // Remove this after that bug is fixed.
            {
                u32 im_count = MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT;
                result = vk_dev_procs.getSwapchainImagesKHR(device_, swapchain, &im_count, p_swapchain_images);
            }

            result = vk_dev_procs.acquireNextImageKHR(
                device_, swapchain, UINT64_MAX, swapchain_image_acquired_semaphore, VK_NULL_HANDLE,
                &acquired_swapchain_image_idx
            );
            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                swapchain_needs_rebuild = true;
                continue;
            }

            // If the swapchain is Suboptimal, we will rebuild it _after_ this frame. This is because
            // acquireNextImageKHR succeeded, so it may have signaled `swapchain_image_acquired_semaphore`;
            // we must not call it again with the same semaphore until the semaphore is unsignaled.
            if (result == VK_SUBOPTIMAL_KHR) swapchain_needs_rebuild = true;
            else {
                assertVk(result);
                swapchain_needs_rebuild = false;
            }
            break;
        };

        if (swapchain_rebuild_count > 0) {

            LOG_F(INFO, "Swapchain rebuilt (%u times).", swapchain_rebuild_count);

            // Before destroying resources, make sure they're not in use.
            result = vk_dev_procs.queueWaitIdle(queue_);
            assertVk(result);


            result = vk_dev_procs.resetCommandPool(device_, command_pool, 0);
            assertVk(result);

            for (u32 i = 0; i < swapchain_image_count; i++) vk_dev_procs.destroyFramebuffer(
                device_, p_simple_render_pass_swapchain_framebuffers[i], NULL
            );

            for (u32 i = 0; i < swapchain_image_count; i++) vk_dev_procs.destroyImageView(
                device_, p_swapchain_image_views[i], NULL
            );


            swapchain_roi = centeredSubregion_16x9(swapchain_extent);

            u32 old_image_count = swapchain_image_count;
            swapchain_image_count = getSwapchainImages(
                device_, swapchain, MAX_EXPECTED_SWAPCHAIN_IMAGE_COUNT, p_swapchain_images
            );
            // I don't feel like handling a changing number of swapchain images.
            // We'd have to change the number of command buffers, fences, semaphores.
            alwaysAssert(swapchain_image_count == old_image_count);

            success = createImageViewsForSwapchain(
                device_, swapchain_image_count, p_swapchain_images, p_swapchain_image_views
            );
            alwaysAssert(success);

            success = createFramebuffersForSwapchain(
                device_, render_pass, swapchain_extent, swapchain_image_count, p_swapchain_image_views,
                p_simple_render_pass_swapchain_framebuffers
            );
            alwaysAssert(success);
        }

        // ---------------------------------------------------------------------------------------------------

        VkCommandBuffer command_buffer = p_command_buffers[acquired_swapchain_image_idx];
        VkFence command_buffer_pending_fence = p_command_buffer_pending_fences[acquired_swapchain_image_idx];
        VkSemaphore render_finished_semaphore = p_render_finished_semaphores[acquired_swapchain_image_idx];


        result = vk_dev_procs.waitForFences(device_, 1, &command_buffer_pending_fence, VK_TRUE, UINT64_MAX);
        assertVk(result);
        result = vk_dev_procs.resetFences(device_, 1, &command_buffer_pending_fence);
        assertVk(result);


        vk_dev_procs.resetCommandBuffer(command_buffer, 0);

        VoxelPipelineVertexShaderPushConstants voxel_pipeline_push_constants {
            .angle_radians = (f32) ( 2.0*M_PI * (1.0/150.0)*fmod((f64)frame_counter, 150.0) ),
        };

        success = recordVoxelCommandBuffer(
            command_buffer,
            render_pass,
            pipelines_.voxel_pipeline,
            pipeline_layouts_.voxel_pipeline_layout,
            swapchain_extent,
            swapchain_roi,
            p_simple_render_pass_swapchain_framebuffers[acquired_swapchain_image_idx],
            &voxel_pipeline_push_constants
        );
        alwaysAssert(success);


        const VkPipelineStageFlags wait_dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        const VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &swapchain_image_acquired_semaphore,
            .pWaitDstStageMask = &wait_dst_stage_mask,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &render_finished_semaphore,
        };
        result = vk_dev_procs.queueSubmit(queue_, 1, &submit_info, command_buffer_pending_fence);
        assertVk(result);


        VkPresentInfoKHR present_info {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_finished_semaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &acquired_swapchain_image_idx,
        };
        result = vk_dev_procs.queuePresentKHR(queue_, &present_info);
        if (result == VK_ERROR_OUT_OF_DATE_KHR or result == VK_SUBOPTIMAL_KHR) swapchain_needs_rebuild = true;
        else assertVk(result);

        frame_counter++;
        glfwPollEvents();
    };


    glfwTerminate();
    exit(0);
}
