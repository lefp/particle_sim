#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cstring>

#include <sys/stat.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <loguru.hpp>

#include "types.hpp"
#include "error_utils.hpp" // TODO rename this to "error_util" to be consistent with "alloc_util" and "math_util"
#include "vk_procs.hpp"
#include "defer.hpp"
#include "alloc_util.hpp"
#include "math_util.hpp"

//
// Global constants ==========================================================================================
//

#define VULKAN_API_VERSION VK_API_VERSION_1_3
#define SWAPCHAIN_FORMAT VK_FORMAT_R8G8B8A8_SRGB
#define SWAPCHAIN_COLOR_SPACE VK_COLOR_SPACE_SRGB_NONLINEAR_KHR

const char* APP_NAME = "an game";

// Use to represent an invalid queue family; can't use -1 (because unsigned) or 0 (because it's valid).
const u32 INVALID_QUEUE_FAMILY_IDX = UINT32_MAX;

//
// Global variables ==========================================================================================
//

VkInstance instance_ = VK_NULL_HANDLE;
VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
VkPhysicalDeviceProperties physical_device_properties_;
u32 queue_family_ = INVALID_QUEUE_FAMILY_IDX;
VkDevice device_ = VK_NULL_HANDLE;
VkQueue queue_ = VK_NULL_HANDLE;

struct {
    VkPipeline temp_triangle_pipeline = VK_NULL_HANDLE;
} pipelines_;

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
/// `device_type_priorities` are interpreted as follows:
///     0 means "do not use".
///     A higher number indicates greater priority.
void selectPhysicalDeviceAndQueueFamily(
    VkInstance instance,
    VkPhysicalDevice* device_out,
    u32* queue_family_out,
    u32fast device_count,
    const VkPhysicalDevice* devices,
    const QueueFamilyRequirements* queue_family_requirements,
    PhysicalDeviceTypePriorities device_type_priorities
) {
    VkPhysicalDevice current_best_device = VK_NULL_HANDLE;
    u8 current_best_device_priority = 0;
    u32 current_best_device_queue_family = INVALID_QUEUE_FAMILY_IDX;

    for (u32fast dev_idx = 0; dev_idx < device_count; dev_idx++) {

        const VkPhysicalDevice device = devices[dev_idx];


        VkPhysicalDeviceProperties device_props {};
        vk_inst_procs.getPhysicalDeviceProperties(device, &device_props);
        const u8 device_priority = device_type_priorities.getPriority(device_props.deviceType);
        if (device_priority <= current_best_device_priority) continue;


        u32 family_count = 0;
        vk_inst_procs.getPhysicalDeviceQueueFamilyProperties(device, &family_count, NULL);
        alwaysAssert(family_count > 0);

        VkQueueFamilyProperties* family_props_list = mallocArray<VkQueueFamilyProperties>(family_count);
        vk_inst_procs.getPhysicalDeviceQueueFamilyProperties(device, &family_count, family_props_list);

        const u32 fam = firstSatisfactoryQueueFamily(
            instance, device, family_count, family_props_list, queue_family_requirements
        );
        if (fam == INVALID_QUEUE_FAMILY_IDX) {
            LOG_F(INFO, "Physical device %lu has no satisfactory queue family.", dev_idx);
            continue;
        }


        current_best_device = device;
        current_best_device_priority = device_priority;
        current_best_device_queue_family = fam;
    }

    *device_out = current_best_device;
    *queue_family_out = current_best_device_queue_family;
}


void initGraphicsUptoQueueCreation(void) {
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

        for (u32 i = 0; i < physical_device_count; i++) {
            VkPhysicalDeviceProperties props;
            vk_inst_procs.getPhysicalDeviceProperties(physical_devices[i], &props);
            LOG_F(INFO, "Found physical device %u: `%s`", i, props.deviceName);
        }


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

        selectPhysicalDeviceAndQueueFamily(
            instance_,
            &physical_device_, &queue_family_,
            physical_device_count, physical_devices,
            &queue_family_requirements, device_type_priorities
        );
        alwaysAssert(physical_device_ != VK_NULL_HANDLE);

        vk_inst_procs.getPhysicalDeviceProperties(physical_device_, &physical_device_properties_);
        LOG_F(INFO, "Selected physical device `%s`.", physical_device_properties_.deviceName);
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


VkPipeline createTrianglePipeline(VkDevice device, VkRenderPass render_pass, u32 subpass) {

    VkShaderModule vertex_shader_module = createShaderModuleFromSpirvFile("build/temp.vert.spv", device);
    alwaysAssert(vertex_shader_module != VK_NULL_HANDLE);
    defer(vk_dev_procs.destroyShaderModule(device, vertex_shader_module, NULL));

    VkShaderModule fragment_shader_module = createShaderModuleFromSpirvFile("build/temp.frag.spv", device);
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
        .cullMode = VK_CULL_MODE_NONE,
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


    const VkPipelineLayoutCreateInfo pipeline_layout_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 0,
        .pSetLayouts = NULL,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = NULL,
    };

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkResult result = vk_dev_procs.createPipelineLayout(device, &pipeline_layout_info, NULL, &pipeline_layout);
    assertVk(result);


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
/// `fallback_extent` is used if the surface doesn't already have a set extent.
/// Doesn't destroy `old_swapchain`; simply retires it. You are responsible for destroying it.
/// `old_swapchain` may be `VK_NULL_HANDLE`.
VkSwapchainKHR createSwapchain(
     VkPhysicalDevice physical_device,
     VkDevice device,
     VkSurfaceKHR surface,
     VkExtent2D fallback_extent,
     u32 queue_family_index,
     VkPresentModeKHR present_mode,
     VkSwapchainKHR old_swapchain
) {

    VkSurfaceCapabilitiesKHR surface_capabilities {};
    VkResult result =
        vk_inst_procs.getPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surface_capabilities);
    assertVk(result);


    // Vk spec 1.3.259, appendix VK_KHR_swapchain, issue 12: suggests using capabilities.minImageCount + 1
    // to guarantee that vkAcquireNextImageKHR is non-blocking when using Mailbox present mode.
    // Idk what effect this has on FIFO present mode, but I'm assuming it doesn't hurt.
    u32 min_image_count = surface_capabilities.minImageCount + 1;
    {
        u32 count_preclamp = min_image_count;
        min_image_count = math::clamp(
            min_image_count,
            surface_capabilities.minImageCount,
            surface_capabilities.maxImageCount
        );
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
    return swapchain;
}


int main(int argc, char** argv) {

    loguru::init(argc, argv);

    int success = glfwInit();
    assertGlfw(success);


    initGraphicsUptoQueueCreation();

    VkRenderPass render_pass = createSimpleRenderPass(device_);
    alwaysAssert(render_pass != VK_NULL_HANDLE);

    const u32 subpass = 0;
    VkPipeline pipeline = createTrianglePipeline(device_, render_pass, subpass);
    alwaysAssert(pipeline != VK_NULL_HANDLE);
    pipelines_.temp_triangle_pipeline = pipeline;

    // TODO remaining work:
    //
    // set up image views and framebuffers
    // set up command buffers
    // vkCmdSetViewport, vkCmdSetScissor


    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // TODO: enable once swapchain resizing is implemented
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // don't initialize OpenGL, because we're using Vulkan
    GLFWwindow* window = glfwCreateWindow(800, 600, "an game", NULL, NULL);
    assertGlfw(window != NULL);

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult result = glfwCreateWindowSurface(instance_, window, NULL, &surface);
    assertVk(result);

    VkSwapchainKHR swapchain = createSwapchain(
        physical_device_,
        device_,
        surface,
        VkExtent2D { 800, 600 },
        queue_family_,
        VK_PRESENT_MODE_FIFO_KHR,
        VK_NULL_HANDLE // old_swapchain
    );
    alwaysAssert(swapchain != VK_NULL_HANDLE);


    glfwPollEvents();
    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();
    };


    glfwTerminate();
    exit(0);
}
