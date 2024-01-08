#include <cerrno>
#include <cinttypes>
#include <cstdio>

// TODO I'd rather not have a dependency on any windowing library in this file. We should be able to
// completely replace GLFW; we're only using:
// 1. glfwGetInstanceProcAddress : replace with dlopen, dlsym
// 2. glfwGetRequiredInstanceExtensions :
//     - We obviously need "VK_KHR_swapchain".
//     - Ask the user what specific window system they're using, via an enum parameter (WAYLAND, XCB, etc).
//         They can probably get that information from GLFW if they're using GLFW. Then enable the appropriate
//         extension: "VK_KHR_wayland_surface", "VK_KHR_xcb_surface", etc
// 3. glfwGetPhysicalDevicePresentationSupport : same thing. Take an enum parameter for the window system, and
//    use vkGetPhysicalDeviceWaylandPresentationSupportKHR or vkGetPhysicalDeviceXlibPresentationSupportKHR,
//    etc
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <loguru/loguru.hpp>
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <VulkanMemoryAllocator/vk_mem_alloc.h>
#include <imgui/imgui_impl_vulkan.h>

#include "types.hpp"
#include "error_util.hpp"
#include "vk_procs.hpp"
#include "alloc_util.hpp"
#include "defer.hpp"
#include "math_util.hpp"
#include "graphics.hpp"

namespace graphics {

using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

//
// Global constants ==========================================================================================
//

#define VULKAN_API_VERSION VK_API_VERSION_1_3

// TODO FIXME:
//     Implement a check to verify that this format is supported. If it isn't, either pick a different format
//     or abort.
const VkFormat SWAPCHAIN_FORMAT = VK_FORMAT_B8G8R8A8_SRGB;
const VkColorSpaceKHR SWAPCHAIN_COLOR_SPACE = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

const VkImageLayout SIMPLE_RENDER_PASS_COLOR_ATTACHMENT_INITIAL_LAYOUT =
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
const VkImageLayout SIMPLE_RENDER_PASS_COLOR_ATTACHMENT_FINAL_LAYOUT =
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // we'll copy the image into a swapchain image

// TODO FIXME:
//     Implement a check to verify that this format is supported. If it isn't, either pick a different format
//     or abort.
const VkFormat DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;
const VkImageLayout DEPTH_IMAGE_LAYOUT = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

// Invalid values for unsigned types, where you can't use -1 and `0` could be valid.
const u32 INVALID_QUEUE_FAMILY_IDX = UINT32_MAX;
const u32 INVALID_PHYSICAL_DEVICE_IDX = UINT32_MAX;
const u32 INVALID_SWAPCHAIN_IMAGE_IDX = UINT32_MAX;
const u32 INVALID_SUBPASS_IDX = UINT32_MAX;

const u32fast MAX_FRAMES_IN_FLIGHT = 2;

const u32fast PHYSICAL_DEVICE_TYPE_COUNT = 5; // number of VK_PHYSICAL_DEVICE_TYPE_xxx variants

//
// Global variables ==========================================================================================
//

static bool initialized_ = false;

static VkInstance instance_ = VK_NULL_HANDLE;
static VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
static VkPhysicalDeviceProperties physical_device_properties_;
static u32 queue_family_ = INVALID_QUEUE_FAMILY_IDX;
static VkDevice device_ = VK_NULL_HANDLE;
static VkQueue queue_ = VK_NULL_HANDLE;

static VkRenderPass simple_render_pass_ = VK_NULL_HANDLE;
static u32 the_only_subpass_ = INVALID_SUBPASS_IDX;

static struct {
    VkPipeline voxel_pipeline = VK_NULL_HANDLE;
    VkPipeline grid_pipeline = VK_NULL_HANDLE;
} pipelines_;

static struct {
    VkPipelineLayout voxel_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayout grid_pipeline_layout = VK_NULL_HANDLE;
} pipeline_layouts_;

// TODO FIXME use a FIFO fallback if this present mode is not supported
static VkPresentModeKHR present_mode_ = VK_PRESENT_MODE_MAILBOX_KHR;

static VmaAllocator vma_allocator_ = NULL;

//
// ===========================================================================================================
//

struct QueueFamilyRequirements {
    VkQueueFlags required_queue_flags;
    bool require_presentation_support;
};

/// Initialize as follows:
///   PhysicalDeviceTypePriorities priorities {};
///   priorities.p[VK_PHYSICAL_DEVICE_TYPE_OTHER] = <your_number_here>
///   priorities.p[VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU] = <your_number_here>
///   ...
struct PhysicalDeviceTypePriorities {
    u8 p[PHYSICAL_DEVICE_TYPE_COUNT];

    u8 getPriority(VkPhysicalDeviceType device_type) {
        alwaysAssert(0 <= device_type && device_type < PHYSICAL_DEVICE_TYPE_COUNT);
        return this->p[device_type];
    }

    static_assert(0 == VK_PHYSICAL_DEVICE_TYPE_OTHER);
    static_assert(1 == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    static_assert(2 == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    static_assert(3 == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU);
    static_assert(4 == VK_PHYSICAL_DEVICE_TYPE_CPU);
    static_assert(5 == PHYSICAL_DEVICE_TYPE_COUNT);
};

struct VoxelPipelineVertexShaderPushConstants {
    mat4 transform;
};

using GridPipelineFragmentShaderPushConstants = CameraInfo;

struct RenderResourcesImpl {
    struct PerFrameResources {

        // Lifetime: same as the lifetime of this RenderResourcesImpl.

        VkCommandBuffer command_buffer;
        VkFence command_buffer_pending_fence;
        // TODO FIXME: Do we need to destroy and recreate this when attaching/detaching from a swapchain,
        // just to be sure that it wasn't left signalled?
        VkSemaphore render_finished_semaphore;

        // Lifetime: as long as this RenderResourcesImpl is attached to a SurfaceImpl.
        // In theory, we only need to destroy them when we attach to a surface of different size than these
        // resources; but we can destroy them for simplicity.

        VkFramebuffer framebuffer;

        // OPTIMIZE:
        // The above resources are accessed at least once per frame.
        // The below resources are only accessed when attaching/detaching from a surface.
        // Maybe store them in a separate array.

        VkImage render_target;
        VkImageView render_target_view;
        VmaAllocation render_target_allocation;

        VkImage depth_buffer;
        VkImageView depth_buffer_view;
        VmaAllocation depth_buffer_allocation;
    };

    VkRenderPass render_pass;
    VkCommandPool command_pool;

    u32 last_used_frame_idx;
    PerFrameResources frame_resources_array[MAX_FRAMES_IN_FLIGHT];


    PerFrameResources* getNextFrameResources(void) {
        u32 this_frame_idx = (this->last_used_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
        this->last_used_frame_idx = this_frame_idx;
        return &this->frame_resources_array[this_frame_idx];
    }
};

struct SurfaceResourcesImpl {

    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;

    // per-image resources
    VkImage* swapchain_images;
    VkSemaphore* swapchain_image_acquired_semaphores;

    VkExtent2D swapchain_extent;
    u32 swapchain_image_count;
    u32 last_used_swapchain_image_acquired_semaphore_idx;

    RenderResourcesImpl* attached_render_resources; // can be NULL if nothing is attached
};

//
// ===========================================================================================================
//

static void _assertGlfw(bool condition, const char* file, int line) {

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


static void _abortIfGlfwError(const char* file, int line) {

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


static bool flagsSubset(VkQueueFlags subset, VkQueueFlags superset) {
    return (subset & superset) == subset;
}


/// If no satisfactory family found, returns `QUEUE_FAMILY_NOT_FOUND`.
static u32 firstSatisfactoryQueueFamily(
      VkInstance instance,
      VkPhysicalDevice device,
      u32 family_count,
      const VkQueueFamilyProperties* family_properties_list,
      const QueueFamilyRequirements* requirements
) {
    for (u32 fam_idx = 0; fam_idx < family_count; fam_idx++) {

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


static void _assertVk(VkResult result, const char* file, int line) {

    if (result == VK_SUCCESS) return;

    LOG_F(
        FATAL, "VkResult is %i, file `%s`, line %i",
        result, file, line
    );
    abort();
}
#define assertVk(result) _assertVk(result, __FILE__, __LINE__)


static void _assertGraphics(Result result, const char* file, int line) {

    if (result == Result::success) return;

    ABORT_F("GraphicsResult is %i, file `%s`, line %i", (int)result, file, line);
}
#define assertGraphics(result) _assertGraphics(result, __FILE__, __LINE__)


/// If no satisfactory device is found, `device_out` is set to `VK_NULL_HANDLE`.
/// You may request a specific physical device using `specific_physical_device_request`.
///     That device is selected iff it exists and satisfies requirements (ignoring `device_type_priorities`).
///     To avoid requesting a specific device, pass `INVALID_PHYSICAL_DEVICE_IDX`.
/// `device_type_priorities` are interpreted as follows:
///     0 means "do not use".
///     A higher number indicates greater priority.
/// Returns the index of the selected device.
static void selectPhysicalDeviceAndQueueFamily(
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
        vk_inst_procs.GetPhysicalDeviceQueueFamilyProperties(device, &family_count, NULL);
        alwaysAssert(family_count > 0);

        VkQueueFamilyProperties* family_props_list = mallocArray(family_count, VkQueueFamilyProperties);
        defer(free(family_props_list));
        vk_inst_procs.GetPhysicalDeviceQueueFamilyProperties(device, &family_count, family_props_list);

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
static void initGraphicsUptoQueueCreation(const char* app_name, const char* specific_named_device_request) {

    if (!glfwVulkanSupported()) ABORT_F("Failed to find Vulkan; do you need to install drivers?");
    vk_base_procs.init((PFN_vkGetInstanceProcAddr)glfwGetInstanceProcAddress);

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
            .pApplicationName = app_name,
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

        VkResult result = vk_base_procs.CreateInstance(&instance_info, NULL, &instance_);
        assertVk(result);

        vk_inst_procs.init(instance_, (PFN_vkGetInstanceProcAddr)glfwGetInstanceProcAddress);
    }

    // Select physical device and queue families -------------------------------------------------------------
    {
        u32 physical_device_count = 0;
        VkResult result = vk_inst_procs.EnumeratePhysicalDevices(instance_, &physical_device_count, NULL);
        assertVk(result);

        if (physical_device_count == 0) ABORT_F("Found no Vulkan devices.");


        VkPhysicalDevice* physical_devices = mallocArray(physical_device_count, VkPhysicalDevice);
        defer(free(physical_devices));

        result = vk_inst_procs.EnumeratePhysicalDevices(instance_, &physical_device_count, physical_devices);
        assertVk(result);


        VkPhysicalDeviceProperties* physical_device_properties_list =
            mallocArray(physical_device_count, VkPhysicalDeviceProperties);
        defer(free(physical_device_properties_list));

        u32 requested_device_idx = INVALID_PHYSICAL_DEVICE_IDX;
        for (u32 dev_idx = 0; dev_idx < physical_device_count; dev_idx++) {

            VkPhysicalDeviceProperties* p_dev_props = &physical_device_properties_list[dev_idx];
            vk_inst_procs.GetPhysicalDeviceProperties(physical_devices[dev_idx], p_dev_props);

            const char* device_name = p_dev_props->deviceName;
            LOG_F(INFO, "Found physical device %" PRIu32 ": `%s`.", dev_idx, device_name);
            if (
                specific_named_device_request != NULL and
                strcmp(specific_named_device_request, device_name) == 0
            ) {
                LOG_F(INFO, "Physical device %" PRIu32 ": name matches requested device.", dev_idx);
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

        PhysicalDeviceTypePriorities device_type_priorities {};
        device_type_priorities.p[VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU] = 1;
        device_type_priorities.p[VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU] = 2;

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

        vk_inst_procs.GetPhysicalDeviceProperties(physical_device_, &physical_device_properties_);
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

        VkResult result = vk_inst_procs.CreateDevice(physical_device_, &device_cinfo, NULL, &device_);
        assertVk(result);

        vk_dev_procs.init(device_, vk_inst_procs.GetDeviceProcAddr);

        // NOTE: Vk Spec 1.3.259:
        //     vkGetDeviceQueue must only be used to get queues that were created with the `flags` parameter
        //     of VkDeviceQueueCreateInfo set to zero.
        vk_dev_procs.GetDeviceQueue(device_, queue_family_, 0, &queue_);
    }
}


/// You own the returned buffer. You may free it using `free()`.
/// On error, either aborts or returns `NULL`.
static void* readEntireFile(const char* fname, size_t* size_out) {
    // OPTIMIZE: Maybe using `open()`, `fstat()`, and `read()` would be faster; because we don't need buffered
    // input, and maybe using `fseek()` to get the file size is unnecessarily slow.

    FILE* file = fopen(fname, "r");
    if (file == NULL) {
        LOG_F(ERROR, "Failed to open file `%s`; errno: `%i`, description: `%s`.", fname, errno, strerror(errno));
        return NULL;
    }

    int result = fseek(file, 0, SEEK_END);
    assertErrno(result == 0);

    size_t file_size;
    {
        long size = ftell(file);
        assertErrno(size >= 0);
        file_size = (size_t)size;
    }

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


static VkShaderModule createShaderModuleFromSpirvFile(const char* spirv_fname, VkDevice device) {

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
    VkResult result = vk_dev_procs.CreateShaderModule(device, &cinfo, NULL, &shader_module);
    assertVk(result);

    return shader_module;
};


static VkRenderPass createSimpleRenderPass(VkDevice device) {
    constexpr u32 attachment_count = 2;
    const VkAttachmentDescription attachment_descriptions[attachment_count] {
        // color attachment
        {
            .format = SWAPCHAIN_FORMAT,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = SIMPLE_RENDER_PASS_COLOR_ATTACHMENT_INITIAL_LAYOUT,
            .finalLayout = SIMPLE_RENDER_PASS_COLOR_ATTACHMENT_FINAL_LAYOUT,
        },
        // depth attachment
        {
            .format = DEPTH_FORMAT,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = DEPTH_IMAGE_LAYOUT,
            .finalLayout = DEPTH_IMAGE_LAYOUT,
        }
    };

    constexpr u32 color_attachment_count = 1;
    const VkAttachmentReference color_attachments[color_attachment_count] {
        {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        }
    };

    const VkAttachmentReference depth_attachment {
        .attachment = 1,
        .layout = DEPTH_IMAGE_LAYOUT,
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
            .pDepthStencilAttachment = &depth_attachment,
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
    VkResult result = vk_dev_procs.CreateRenderPass(device, &render_pass_info, NULL, &render_pass);
    assertVk(result);

    return render_pass;
}


static VkPipeline createVoxelPipeline(
    VkDevice device,
    VkRenderPass render_pass,
    u32 subpass,
    VkPipelineLayout* pipeline_layout_out
) {

    VkShaderModule vertex_shader_module = createShaderModuleFromSpirvFile("build/voxel.vert.spv", device);
    alwaysAssert(vertex_shader_module != VK_NULL_HANDLE);
    defer(vk_dev_procs.DestroyShaderModule(device, vertex_shader_module, NULL));

    VkShaderModule fragment_shader_module = createShaderModuleFromSpirvFile("build/voxel.frag.spv", device);
    alwaysAssert(fragment_shader_module != VK_NULL_HANDLE);
    defer(vk_dev_procs.DestroyShaderModule(device, fragment_shader_module, NULL));

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
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
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
    VkResult result = vk_dev_procs.CreatePipelineLayout(device, &pipeline_layout_info, NULL, &pipeline_layout);
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
    result = vk_dev_procs.CreateGraphicsPipelines(
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


static VkPipeline createGridPipeline(
    VkDevice device,
    VkRenderPass render_pass,
    u32 subpass,
    VkPipelineLayout* pipeline_layout_out
) {

    VkShaderModule vertex_shader_module = createShaderModuleFromSpirvFile("build/grid.vert.spv", device);
    alwaysAssert(vertex_shader_module != VK_NULL_HANDLE);
    defer(vk_dev_procs.DestroyShaderModule(device, vertex_shader_module, NULL));

    VkShaderModule fragment_shader_module = createShaderModuleFromSpirvFile("build/grid.frag.spv", device);
    alwaysAssert(fragment_shader_module != VK_NULL_HANDLE);
    defer(vk_dev_procs.DestroyShaderModule(device, fragment_shader_module, NULL));

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
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = {},
        .back = {},
        .minDepthBounds = 0.0,
        .maxDepthBounds = 1.0,
    };


    // VK_BLEND_OP_ADD :
    //    alpha: a_new = a_src * srcAlphaBlendFactor + a_dst * dstAlphaBlendFactor
    //    color: C_new = C_src * srcColorBlendFactor + C_dst * dstColorBlendFactor
    const VkPipelineColorBlendAttachmentState color_blend_attachment_info {
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
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
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = sizeof(GridPipelineFragmentShaderPushConstants),
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
    VkResult result = vk_dev_procs.CreatePipelineLayout(device, &pipeline_layout_info, NULL, &pipeline_layout);
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
    result = vk_dev_procs.CreateGraphicsPipelines(
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
/// `swapchain_out` and `extent_out` must not be NULL.
static Result createSwapchain(
     VkPhysicalDevice physical_device,
     VkDevice device,
     VkSurfaceKHR surface,
     VkExtent2D fallback_extent,
     u32 queue_family_index,
     VkPresentModeKHR present_mode,
     VkSwapchainKHR old_swapchain,
     VkSwapchainKHR* swapchain_out,
     VkExtent2D* extent_out
) {

    VkSurfaceCapabilitiesKHR surface_capabilities {};
    VkResult result = vk_inst_procs.GetPhysicalDeviceSurfaceCapabilitiesKHR(
        physical_device, surface, &surface_capabilities
    );
    assertVk(result);


    // Vk spec 1.3.259, appendix VK_KHR_swapchain, issue 12: suggests using capabilities.minImageCount + 1
    // to guarantee that vkAcquireNextImageKHR is non-blocking when using Mailbox present mode.
    // In FIFO mode, I don't expect this to have any negative effect, as long as we don't render too many
    // frames in advance (which would cause noticeable input latency).
    u32 min_image_count = surface_capabilities.minImageCount + 1;
    {
        u32 count_preclamp = min_image_count;
        min_image_count = math::max(min_image_count, surface_capabilities.minImageCount);
        if (surface_capabilities.maxImageCount != 0) {
            min_image_count = math::min(min_image_count, surface_capabilities.maxImageCount);
        }

        if (min_image_count != count_preclamp) LOG_F(
            WARNING,
            "Min swapchain image count clamped from %" PRIu32 " to %" PRIu32 ", to fit surface limits.",
            count_preclamp, min_image_count
        );
    }
    LOG_F(INFO, "Will request minImageCount=%" PRIu32 " for swapchain creation.", min_image_count);


    // Vk spec 1.3.234:
    //     On some platforms, it is normal that maxImageExtent may become (0, 0), for example when the window
    //     is minimized. In such a case, it is not possible to create a swapchain due to the Valid Usage
    //     requirements.
    const VkExtent2D max_extent = surface_capabilities.maxImageExtent;
    if (max_extent.width == 0 or max_extent.height == 0) {
        LOG_F(INFO, "Aborting swapchain build: SurfaceCapabilities::maxImageExtent contains a 0.");
        return Result::error_window_size_zero;
    }

    VkExtent2D extent = surface_capabilities.currentExtent;
    // Vk spec 1.3.234:
    //     currentExtent is the current width and height of the surface, or the special value (0xFFFFFFFF,
    //     0xFFFFFFFF) indicating that the surface size will be determined by the extent of a swapchain
    //     targeting the surface.
    if (extent.width == 0xFF'FF'FF'FF and extent.height == 0xFF'FF'FF'FF) {

        LOG_F(INFO, "Surface currentExtent is (0xFFFFFFFF, 0xFFFFFFFF); using fallback extent.");
        if (fallback_extent.width == 0 or fallback_extent.height == 0) {
            LOG_F(INFO, "Aborting swapchain build: `fallback_extent` contains a 0.");
            return Result::error_window_size_zero;
        }

        const VkExtent2D min_extent = surface_capabilities.minImageExtent;

        extent.width = math::clamp(fallback_extent.width, min_extent.width, max_extent.width);
        extent.height = math::clamp(fallback_extent.height, min_extent.height, max_extent.height);
        if (extent.width != fallback_extent.width or extent.height != fallback_extent.height) LOG_F(
            WARNING,
            "Adjusted fallback swapchain extent (%" PRIu32 ", %" PRIu32 ") to (%" PRIu32 ", %" PRIu32 "), to fit surface limits.",
            fallback_extent.width, fallback_extent.height, extent.width, extent.height
        );
    }


    alwaysAssert(surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT);


    VkSwapchainCreateInfoKHR swapchain_info {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = min_image_count,
        .imageFormat = SWAPCHAIN_FORMAT,
        .imageColorSpace = SWAPCHAIN_COLOR_SPACE,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
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
    result = vk_dev_procs.CreateSwapchainKHR(device, &swapchain_info, NULL, &swapchain);
    assertVk(result);

    LOG_F(INFO, "Built swapchain %p.", swapchain);
    *swapchain_out = swapchain;
    *extent_out = extent;
    return Result::success;
}


/*
/// Returns whether it succeeded.
static bool createImageViewsForSwapchain(
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

        VkResult result = vk_dev_procs.CreateImageView(device, &image_view_info, NULL, p_image_view);
        assertVk(result);
    }

    return true;
}
*/


static void createPerSwapchainImageSurfaceResources(
    VkSwapchainKHR swapchain,
    u32* image_count_out,
    VkImage** images_out,
    VkSemaphore** image_acquired_semaphores_out
) {

    u32 image_count = 0;
    VkResult result = vk_dev_procs.GetSwapchainImagesKHR(device_, swapchain, &image_count, NULL);
    assertVk(result);

    LOG_F(INFO, "Got %" PRIu32 " images from swapchain %p.", image_count, swapchain);


    VkImage* swapchain_images = mallocArray(image_count, VkImage);
    result = vk_dev_procs.GetSwapchainImagesKHR(device_, swapchain, &image_count, swapchain_images);
    assertVk(result);


    VkSemaphore* swapchain_image_acquired_semaphores = mallocArray(image_count, VkSemaphore);

    VkSemaphoreCreateInfo semaphore_info { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for (u32 im_idx = 0; im_idx < image_count; im_idx++) {
        result = vk_dev_procs.CreateSemaphore(
            device_, &semaphore_info, NULL, &swapchain_image_acquired_semaphores[im_idx]
        );
        assertVk(result);
    }


    *image_count_out = image_count;
    *images_out = swapchain_images;
    *image_acquired_semaphores_out = swapchain_image_acquired_semaphores;
}

/// Does not synchronize.
/// You are responsible for ensuring the resources are not in use (e.g. via vkDeviceWaitIdle).
static void destroyPerSwapchainImageSurfaceResources(
    u32 image_count,
    VkImage** pp_images,
    VkSemaphore** pp_image_acquired_semaphores
) {
    assert(pp_images != NULL);
    assert(pp_image_acquired_semaphores != NULL);

    VkSemaphore* p_image_acquired_semaphores = *pp_image_acquired_semaphores;

    for (u32 im_idx = 0; im_idx < image_count; im_idx++) {
        vk_dev_procs.DestroySemaphore(device_, p_image_acquired_semaphores[im_idx], NULL);
    }
    free(*pp_image_acquired_semaphores);

    free(*pp_images);
}


// TODO remove
/*
/// Returns whether it succeeded.
static bool createFramebuffersForSwapchain(
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

        VkResult result = vk_dev_procs.CreateFramebuffer(device, &framebuffer_info, NULL, p_framebuffer);
        assertVk(result);
    }

    return true;
}
*/


/// Does not call `BeginCommandBuffer` and `EndCommandBuffer`; you are responsible for doing that.
/// Returns `true` if successful.
static bool recordCommandBuffer(
    VkCommandBuffer command_buffer,
    VkRenderPass render_pass,
    VkPipeline voxel_pipeline,
    VkPipelineLayout voxel_pipeline_layout,
    VkPipeline grid_pipeline,
    VkPipelineLayout grid_pipeline_layout,
    VkExtent2D swapchain_extent,
    VkRect2D swapchain_roi,
    VkFramebuffer framebuffer,
    const VoxelPipelineVertexShaderPushConstants* voxel_pipeline_push_constants,
    const GridPipelineFragmentShaderPushConstants* grid_pipeline_push_constants,
    ImDrawData* imgui_draw_data
) {

    // TODO replace magic number with some named constant (it's the number of render pass attachments with
    // loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR)
    constexpr u32 clear_value_count = 2;
    VkClearValue clear_values[clear_value_count] {
        { .color = VkClearColorValue { .float32 = {0, 0, 0, 1} } },
        { .depthStencil = VkClearDepthStencilValue { .depth = 1 } },
    };
    VkRenderPassBeginInfo render_pass_begin_info {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = render_pass,
        .framebuffer = framebuffer,
        .renderArea = VkRect2D { .offset = {0, 0}, .extent = swapchain_extent },
        .clearValueCount = clear_value_count,
        .pClearValues = clear_values,
    };
    vk_dev_procs.CmdBeginRenderPass(
        command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE
    );

    const VkViewport viewport {
        .x = (f32)swapchain_roi.offset.x,
        .y = (f32)swapchain_roi.offset.y,
        .width = (f32)swapchain_roi.extent.width,
        .height = (f32)swapchain_roi.extent.height,
        .minDepth = 0,
        .maxDepth = 1,
    };
    vk_dev_procs.CmdSetViewport(command_buffer, 0, 1, &viewport);
    vk_dev_procs.CmdSetScissor(command_buffer, 0, 1, &swapchain_roi);


    vk_dev_procs.CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxel_pipeline);

    vk_dev_procs.CmdPushConstants(
        command_buffer, voxel_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
        sizeof(*voxel_pipeline_push_constants), voxel_pipeline_push_constants
    );

    vk_dev_procs.CmdDraw(command_buffer, 36, 1, 0, 0);


    vk_dev_procs.CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, grid_pipeline);

    vk_dev_procs.CmdPushConstants(
        command_buffer, grid_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        sizeof(*grid_pipeline_push_constants), grid_pipeline_push_constants
    );

    vk_dev_procs.CmdDraw(command_buffer, 6, 1, 0, 0);


    if (imgui_draw_data != NULL) ImGui_ImplVulkan_RenderDrawData(imgui_draw_data, command_buffer);

    vk_dev_procs.CmdEndRenderPass(command_buffer);

    return true;
}


static void imguiVkResultCheckCallback(VkResult result) {

    if (result == VK_SUCCESS) return;

    // TODO what if it's a recoverable result, similarly to VK_SUBOPTIMAL_KHR?
    ABORT_F("Imgui encountered VkResult %i.", result);
}


static PFN_vkVoidFunction imguiLoadVkProcCallback(const char* proc_name, void* user_data) {
    (void)user_data;

    assert(initialized_);

    // NOTE I assume here, without a strong reason, that ImGui will only attempt to get procedures that can be
    // obtained via vkGetDeviceProcAddr.
    return vk_base_procs.GetInstanceProcAddr(instance_, proc_name);
}


extern bool initImGuiVulkanBackend(void) {

    alwaysAssert(initialized_);

    bool success = ImGui_ImplVulkan_LoadFunctions(imguiLoadVkProcCallback, NULL);
    alwaysAssert(success);

    // This descriptor pool initialization was pretty much copied from imgui's `example_glfw_vulkan`
    // TODO do we actually need this?
    VkDescriptorPoolSize descriptor_pool_size {
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
    };
    VkDescriptorPoolCreateInfo descriptor_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &descriptor_pool_size,
    };

    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkResult result = vk_dev_procs.CreateDescriptorPool(
        device_, &descriptor_pool_info, NULL, &descriptor_pool
    );
    assertVk(result);

    ImGui_ImplVulkan_InitInfo imgui_vk_init_info {
        .Instance = instance_,
        .PhysicalDevice = physical_device_,
        .Device = device_,
        .QueueFamily = queue_family_,
        .Queue = queue_,
        .PipelineCache = VK_NULL_HANDLE,
        .DescriptorPool = descriptor_pool,
        .Subpass = the_only_subpass_,
        // Imgui asserts >= 2. Pretty sure the actual number doesn't matter, because
        // Imgui doesn't even use this, as long as we don't use its swapchain creation helpers.
        .MinImageCount = 2,
        .ImageCount = MAX_FRAMES_IN_FLIGHT, // NOTE I _think_ this is right, but not sure
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .CheckVkResultFn = imguiVkResultCheckCallback,
    };
    bool imgui_result = ImGui_ImplVulkan_Init(&imgui_vk_init_info, simple_render_pass_);
    return imgui_result;
}


extern void init(const char* app_name, const char* specific_named_device_request) {

    // TODO remaining work:
    // Set up validation layer debug logging thing, to log their messages as loguru messages. This way they
    // will be logged if we're logging to a file.

    initGraphicsUptoQueueCreation(app_name, specific_named_device_request);


    VmaVulkanFunctions vma_vulkan_functions {
        .vkGetInstanceProcAddr = vk_base_procs.GetInstanceProcAddr,
        .vkGetDeviceProcAddr = vk_inst_procs.GetDeviceProcAddr,
    };

    VmaAllocatorCreateInfo vma_allocator_info {
        .physicalDevice = physical_device_,
        .device = device_,
        .pVulkanFunctions = &vma_vulkan_functions,
        .instance = instance_,
        .vulkanApiVersion = VULKAN_API_VERSION,
    };
    VkResult result = vmaCreateAllocator(&vma_allocator_info, &vma_allocator_);
    assertVk(result);


    VkRenderPass render_pass = createSimpleRenderPass(device_);
    alwaysAssert(render_pass != VK_NULL_HANDLE);
    simple_render_pass_ = render_pass;


    the_only_subpass_ = 0;
    {
        VkPipeline pipeline;

        pipeline = createVoxelPipeline(
            device_, render_pass, the_only_subpass_, &pipeline_layouts_.voxel_pipeline_layout
        );
        alwaysAssert(pipeline != VK_NULL_HANDLE);
        pipelines_.voxel_pipeline = pipeline;

        pipeline = createGridPipeline(
            device_, render_pass, the_only_subpass_, &pipeline_layouts_.grid_pipeline_layout
        );
        alwaysAssert(pipeline != VK_NULL_HANDLE);
        pipelines_.grid_pipeline = pipeline;
    }

    initialized_ = true;
}


extern Result createSurfaceResources(
    VkSurfaceKHR surface,
    VkExtent2D fallback_window_size,
    SurfaceResources* surface_resources_out
) {

    SurfaceResourcesImpl* p_resources = (SurfaceResourcesImpl*)calloc(1, sizeof(SurfaceResourcesImpl));
    assertErrno(p_resources != NULL);


    p_resources->surface = surface;
    p_resources->attached_render_resources = NULL;
    p_resources->last_used_swapchain_image_acquired_semaphore_idx = 0;


    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D swapchain_extent {};
    Result res = createSwapchain(
        physical_device_, device_, surface, fallback_window_size, queue_family_, present_mode_,
        VK_NULL_HANDLE, // old_swapchain
        &swapchain, &swapchain_extent
    );
    if (res == Result::error_window_size_zero) {
        free(p_resources);
        return res;
    };
    assertGraphics(res);

    p_resources->swapchain = swapchain;
    p_resources->swapchain_extent = swapchain_extent;


    createPerSwapchainImageSurfaceResources(
        swapchain,
        &p_resources->swapchain_image_count,
        &p_resources->swapchain_images,
        &p_resources->swapchain_image_acquired_semaphores
    );


    surface_resources_out->impl = p_resources;
    return Result::success;
}


/// `renderer` must not currently have any surface attached.
/// `surface` must not currently have any renderer attached.
extern void attachSurfaceToRenderer(SurfaceResources surface, RenderResources renderer) {

    VkResult result;

    SurfaceResourcesImpl* p_surface_resources = (SurfaceResourcesImpl*)surface.impl;
    RenderResourcesImpl* p_render_resources = (RenderResourcesImpl*)renderer.impl;

    LOG_F(INFO, "Attaching surface %p to renderer %p.", p_surface_resources, p_render_resources);

    if (p_surface_resources->attached_render_resources != NULL) {
        ABORT_F("Attempt to attach surface to renderer, but surface is already attached to a renderer.");
    }
    p_surface_resources->attached_render_resources = p_render_resources;

    u32 swapchain_image_count = p_surface_resources->swapchain_image_count;
    alwaysAssert(0 < swapchain_image_count);


    VkExtent2D swapchain_extent = p_surface_resources->swapchain_extent;


    result = vk_dev_procs.QueueWaitIdle(queue_); // ensure none of the command buffers are busy
    assertVk(result);

    for (u32 frame_idx = 0; frame_idx < MAX_FRAMES_IN_FLIGHT; frame_idx++) {

        RenderResourcesImpl::PerFrameResources* this_frame_resources =
            &p_render_resources->frame_resources_array[frame_idx];


        VkImageCreateInfo color_image_info {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = SWAPCHAIN_FORMAT,
            .extent = VkExtent3D { swapchain_extent.width, swapchain_extent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &queue_family_,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VmaAllocationCreateInfo color_image_alloc_info {
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        };
        result = vmaCreateImage(
            vma_allocator_, &color_image_info, &color_image_alloc_info,
            &this_frame_resources->render_target, &this_frame_resources->render_target_allocation,
            NULL // pAllocationInfo, an output parameter
        );
        assertVk(result);

        VkImageCreateInfo depth_image_info {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = DEPTH_FORMAT,
            .extent = VkExtent3D { swapchain_extent.width, swapchain_extent.height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &queue_family_,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VmaAllocationCreateInfo depth_image_alloc_info {
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        };
        result = vmaCreateImage(
            vma_allocator_, &depth_image_info, &depth_image_alloc_info,
            &this_frame_resources->depth_buffer,
            &this_frame_resources->depth_buffer_allocation,
            NULL // pAllocationInfo, an output parameter
        );
        assertVk(result);


        VkCommandBuffer command_buffer = this_frame_resources->command_buffer;

        VkCommandBufferBeginInfo cmd_buf_begin_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        result = vk_dev_procs.BeginCommandBuffer(command_buffer, &cmd_buf_begin_info);
        assertVk(result);

        constexpr u32 image_barrier_count = 2;
        VkImageMemoryBarrier image_barriers[image_barrier_count] {
            // color image
            {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_NONE,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = SIMPLE_RENDER_PASS_COLOR_ATTACHMENT_INITIAL_LAYOUT,
                .srcQueueFamilyIndex = queue_family_,
                .dstQueueFamilyIndex = queue_family_,
                .image = this_frame_resources->render_target,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            },
            // depth image
            {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_NONE,
                .dstAccessMask =
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = DEPTH_IMAGE_LAYOUT,
                .srcQueueFamilyIndex = queue_family_,
                .dstQueueFamilyIndex = queue_family_,
                .image = this_frame_resources->depth_buffer,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            },
        };

        vk_dev_procs.CmdPipelineBarrier(
            command_buffer,
            // TOP_OF_PIPE_BIT "specifies no stage of execution when specified in the first scope"
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, // srcStageMask
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // dstStageMask
            0, // dependencyFlags
            0, // memoryBarrierCount
            NULL, // pMemoryBarriers
            0, // bufferMemoryBarrierCount
            NULL, // pBufferMemoryBarriers
            image_barrier_count, // imageMemoryBarrierCount
            image_barriers // pImageMemoryBarriers
        );

        result = vk_dev_procs.EndCommandBuffer(command_buffer);
        assertVk(result);

        VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
        };
        result = vk_dev_procs.QueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE);
        assertVk(result);


        VkImageViewCreateInfo color_image_view_info {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = this_frame_resources->render_target,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = SWAPCHAIN_FORMAT,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_R,
                .g = VK_COMPONENT_SWIZZLE_G,
                .b = VK_COMPONENT_SWIZZLE_B,
                .a = VK_COMPONENT_SWIZZLE_A,
            },
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        result = vk_dev_procs.CreateImageView(
            device_,
            &color_image_view_info,
            NULL,
            &this_frame_resources->render_target_view
        );
        assertVk(result);

        VkImageViewCreateInfo depth_image_view_info {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = this_frame_resources->depth_buffer,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = DEPTH_FORMAT,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_R,
                .g = VK_COMPONENT_SWIZZLE_G,
                .b = VK_COMPONENT_SWIZZLE_B,
                .a = VK_COMPONENT_SWIZZLE_A,
            },
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        result = vk_dev_procs.CreateImageView(
            device_,
            &depth_image_view_info,
            NULL,
            &this_frame_resources->depth_buffer_view
        );
        assertVk(result);


        constexpr u32 attachment_count = 2;
        const VkImageView attachments[attachment_count] {
            this_frame_resources->render_target_view,
            this_frame_resources->depth_buffer_view,
        };
        const VkFramebufferCreateInfo framebuffer_info {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = p_render_resources->render_pass,
            .attachmentCount = 2,
            .pAttachments = attachments,
            .width = swapchain_extent.width,
            .height = swapchain_extent.height,
            .layers = 1,
        };
        VkFramebuffer* p_framebuffer = &this_frame_resources->framebuffer;
        result = vk_dev_procs.CreateFramebuffer(device_, &framebuffer_info, NULL, p_framebuffer);
        assertVk(result);
    }

    // wait for any command buffers we just submitted, such as barriers for image layout transitions
    result = vk_dev_procs.QueueWaitIdle(queue_);
    assertVk(result);
}

extern void detachSurfaceFromRenderer(SurfaceResources surface, RenderResources renderer) {

    SurfaceResourcesImpl* p_surface_resources = (SurfaceResourcesImpl*)surface.impl;
    RenderResourcesImpl* p_render_resources = (RenderResourcesImpl*)renderer.impl;

    LOG_F(INFO, "Detaching surface %p from renderer %p.", p_surface_resources, p_render_resources);

    alwaysAssert(p_surface_resources->attached_render_resources == p_render_resources);
    p_surface_resources->attached_render_resources = NULL;


    VkResult result = vk_dev_procs.QueueWaitIdle(queue_);
    assertVk(result);

    for (u32 frame_idx = 0; frame_idx < MAX_FRAMES_IN_FLIGHT; frame_idx++) {

        RenderResourcesImpl::PerFrameResources* this_frame_resources =
            &p_render_resources->frame_resources_array[frame_idx];

        vk_dev_procs.DestroyFramebuffer(device_, this_frame_resources->framebuffer, NULL);
        this_frame_resources->framebuffer = VK_NULL_HANDLE;

        vk_dev_procs.DestroyImageView(device_, this_frame_resources->render_target_view, NULL);
        vmaDestroyImage(
            vma_allocator_,
            this_frame_resources->render_target,
            this_frame_resources->render_target_allocation
        );
        this_frame_resources->render_target_view = VK_NULL_HANDLE;
        this_frame_resources->render_target_allocation = VMA_NULL;

        vk_dev_procs.DestroyImageView(device_, this_frame_resources->depth_buffer_view, NULL);
        vmaDestroyImage(
            vma_allocator_,
            this_frame_resources->depth_buffer,
            this_frame_resources->depth_buffer_allocation
        );
        this_frame_resources->depth_buffer = VK_NULL_HANDLE;
        this_frame_resources->depth_buffer_allocation = VMA_NULL;
    }
}


/// Use if a window resize has caused the resources to be out-of-date.
extern Result updateSurfaceResources(
    SurfaceResources surface_resources,
    VkExtent2D fallback_window_size
) {

    SurfaceResourcesImpl* p_surface_resources = (SurfaceResourcesImpl*)surface_resources.impl;


    VkSwapchainKHR old_swapchain = p_surface_resources->swapchain;
    VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;

    Result res = createSwapchain(
        physical_device_, device_, p_surface_resources->surface, fallback_window_size, queue_family_,
        present_mode_, old_swapchain, &new_swapchain, &p_surface_resources->swapchain_extent
    );
    if (res == Result::error_window_size_zero) return res;
    else assertGraphics(res);

    p_surface_resources->swapchain = new_swapchain;


    // Destory out-of-date resources.
    {
        // Make sure they're not in use.
        VkResult result = vk_dev_procs.QueueWaitIdle(queue_);
        assertVk(result);

        destroyPerSwapchainImageSurfaceResources(
            p_surface_resources->swapchain_image_count,
            &p_surface_resources->swapchain_images,
            &p_surface_resources->swapchain_image_acquired_semaphores
        );

        vk_dev_procs.DestroySwapchainKHR(device_, old_swapchain, NULL);
    }

    // Recreate resources.
    {
        createPerSwapchainImageSurfaceResources(
            new_swapchain,
            &p_surface_resources->swapchain_image_count,
            &p_surface_resources->swapchain_images,
            &p_surface_resources->swapchain_image_acquired_semaphores
        );

        p_surface_resources->last_used_swapchain_image_acquired_semaphore_idx = 0;
    }

    RenderResourcesImpl* p_render_resources = p_surface_resources->attached_render_resources;
    if (p_render_resources != NULL) { // a renderer is attached
        // OPTIMIZE: defaulted to nuclear option here; consider whether we want to do this more efficiently.
        detachSurfaceFromRenderer(surface_resources, RenderResources { .impl = p_render_resources });
        attachSurfaceToRenderer(surface_resources, RenderResources { .impl = p_render_resources });
    }

    return Result::success;
}


extern Result createRenderer(RenderResources* render_resources_out) {

    RenderResourcesImpl* p_render_resources = (RenderResourcesImpl*)calloc(1, sizeof(RenderResourcesImpl));
    assertErrno(p_render_resources != NULL);


    p_render_resources->render_pass = simple_render_pass_;


    const VkCommandPoolCreateInfo command_pool_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family_,
    };

    VkResult result = vk_dev_procs.CreateCommandPool(
        device_, &command_pool_info, NULL, &p_render_resources->command_pool
    );
    assertVk(result);


    VkCommandBuffer p_command_buffers[MAX_FRAMES_IN_FLIGHT] {};
    {
        VkCommandBufferAllocateInfo cmd_buf_alloc_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = p_render_resources->command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
        };
        result = vk_dev_procs.AllocateCommandBuffers(device_, &cmd_buf_alloc_info, p_command_buffers);
        assertVk(result);
    }

    for (u32fast frame_idx = 0; frame_idx < MAX_FRAMES_IN_FLIGHT; frame_idx++) {

        p_render_resources->frame_resources_array[frame_idx].command_buffer = p_command_buffers[frame_idx];

        VkFence* p_fence = &p_render_resources->frame_resources_array[frame_idx].command_buffer_pending_fence;
        {
            VkFenceCreateInfo fence_info {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .flags = VK_FENCE_CREATE_SIGNALED_BIT,
            };
            result = vk_dev_procs.CreateFence(device_, &fence_info, NULL, p_fence);
            assertVk(result);
        }

        VkSemaphore* p_semaphore = &p_render_resources->frame_resources_array[frame_idx].render_finished_semaphore;
        {
            VkSemaphoreCreateInfo semaphore_info { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            result = vk_dev_procs.CreateSemaphore(device_, &semaphore_info, NULL, p_semaphore);
            assertVk(result);
        }
    }
    p_render_resources->last_used_frame_idx = 0;


    render_resources_out->impl = p_render_resources;
    return Result::success;
}


RenderResult render(
    SurfaceResources surface,
    VkRect2D window_subregion,
    const mat4* world_to_screen_transform,
    const CameraInfo* camera_info,
    ImDrawData* imgui_draw_data
) {

    VkResult result;

    SurfaceResourcesImpl* p_surface_resources = (SurfaceResourcesImpl*)surface.impl;
    RenderResourcesImpl* p_render_resources = p_surface_resources->attached_render_resources;

    if (p_render_resources == NULL) ABORT_F("render(): Surface is not attached to a renderer.");


    u32 acquired_swapchain_image_idx = INVALID_SWAPCHAIN_IMAGE_IDX;
    VkSemaphore swapchain_image_acquired_semaphore = VK_NULL_HANDLE;
    {
        u32 im_acquired_semaphore_idx =
            (p_surface_resources->last_used_swapchain_image_acquired_semaphore_idx + 1)
            % p_surface_resources->swapchain_image_count;

        swapchain_image_acquired_semaphore =
            p_surface_resources->swapchain_image_acquired_semaphores[im_acquired_semaphore_idx];

        result = vk_dev_procs.AcquireNextImageKHR(
            device_,
            p_surface_resources->swapchain,
            UINT64_MAX,
            swapchain_image_acquired_semaphore,
            VK_NULL_HANDLE,
            &acquired_swapchain_image_idx
        );

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            LOG_F(INFO, "acquireNextImageKHR returned VK_ERROR_OUT_OF_DATE_KHR. `render()` returning early.");
            return RenderResult::error_surface_resources_out_of_date;
        }
        else if (result == VK_SUBOPTIMAL_KHR) {} // do nothing; we'll handle it after vkQueuePresentKHR
        else assertVk(result);

        p_surface_resources->last_used_swapchain_image_acquired_semaphore_idx = im_acquired_semaphore_idx;
    }


    RenderResourcesImpl::PerFrameResources* this_frame_resources = p_render_resources->getNextFrameResources();


    const VkFence command_buffer_pending_fence = this_frame_resources->command_buffer_pending_fence;

    result = vk_dev_procs.WaitForFences(device_, 1, &command_buffer_pending_fence, VK_TRUE, UINT64_MAX);
    assertVk(result);

    result = vk_dev_procs.ResetFences(device_, 1, &command_buffer_pending_fence);
    assertVk(result);


    VkCommandBuffer command_buffer = this_frame_resources->command_buffer;

    vk_dev_procs.ResetCommandBuffer(command_buffer, 0);


    VoxelPipelineVertexShaderPushConstants voxel_pipeline_push_constants {
        // OPTIMIZE: you're copying 64 bytes here (mat4)
        .transform = *world_to_screen_transform
    };


    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    result = vk_dev_procs.BeginCommandBuffer(command_buffer, &begin_info);
    assertVk(result);
    {
        // TODO maybe we shouldn't hardcode this, if we're doing the whole "attached renderer" thing?
        // Maybe have a function pointer in the renderer or something to the appropriate Render function. Idk,
        // this is getting kinda weird. Maybe we should just ditch the whole generic crap.
        bool success = recordCommandBuffer(
            command_buffer,
            p_render_resources->render_pass,
            pipelines_.voxel_pipeline,
            pipeline_layouts_.voxel_pipeline_layout,
            pipelines_.grid_pipeline,
            pipeline_layouts_.grid_pipeline_layout,
            p_surface_resources->swapchain_extent,
            window_subregion,
            this_frame_resources->framebuffer,
            &voxel_pipeline_push_constants,
            camera_info,
            imgui_draw_data
        );
        alwaysAssert(success);


        // transition swapchain image layout to transfer_dst
        {
            VkImageMemoryBarrier swapchain_image_barrier {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_NONE,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .srcQueueFamilyIndex = queue_family_,
                .dstQueueFamilyIndex = queue_family_,
                .image = p_surface_resources->swapchain_images[acquired_swapchain_image_idx],
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
            vk_dev_procs.CmdPipelineBarrier(
                command_buffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, // srcStageMask
                VK_PIPELINE_STAGE_TRANSFER_BIT, // dstStageMask
                0, // dependencyFlags
                0, // memoryBarrierCount
                NULL, // pMemoryBarriers
                0, // bufferMemoryBarrierCount
                NULL, // pBufferMemoryBarriers
                1, // imageMemoryBarrierCount
                &swapchain_image_barrier // pImageMemoryBarriers
            );
        }

        {
            VkExtent2D swapchain_extent = p_surface_resources->swapchain_extent;
            VkImageCopy image_copy_regions {
                .srcSubresource = VkImageSubresourceLayers {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .srcOffset = VkOffset3D { 0, 0, 0 },
                .dstSubresource = VkImageSubresourceLayers {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .dstOffset = VkOffset3D { 0, 0, 0 },
                .extent = VkExtent3D {
                    .width = swapchain_extent.width,
                    .height = swapchain_extent.height,
                    .depth = 1,
                },
            };
            vk_dev_procs.CmdCopyImage(
                command_buffer,
                this_frame_resources->render_target, // srcImage
                SIMPLE_RENDER_PASS_COLOR_ATTACHMENT_FINAL_LAYOUT, // srcImageLayout
                p_surface_resources->swapchain_images[acquired_swapchain_image_idx], // dstImage
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, // dstImageLayout
                1,
                &image_copy_regions
            );
        }

        // transition swapchain image to present_src
        {
            VkImageMemoryBarrier swapchain_image_barrier {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_NONE, // TODO FIXME is this right?
                .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                .srcQueueFamilyIndex = queue_family_,
                .dstQueueFamilyIndex = queue_family_,
                .image = p_surface_resources->swapchain_images[acquired_swapchain_image_idx],
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
            vk_dev_procs.CmdPipelineBarrier(
                command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, // srcStageMask
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, // dstStageMask // TODO FIXME is this right?
                0, // dependencyFlags
                0, // memoryBarrierCount
                NULL, // pMemoryBarriers
                0, // bufferMemoryBarrierCount
                NULL, // pBufferMemoryBarriers
                1, // imageMemoryBarrierCount
                &swapchain_image_barrier // pImageMemoryBarriers
            );
        }

        VkImageMemoryBarrier color_image_barrier {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = SIMPLE_RENDER_PASS_COLOR_ATTACHMENT_FINAL_LAYOUT,
            .newLayout = SIMPLE_RENDER_PASS_COLOR_ATTACHMENT_INITIAL_LAYOUT,
            .srcQueueFamilyIndex = queue_family_,
            .dstQueueFamilyIndex = queue_family_,
            .image = this_frame_resources->render_target,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        vk_dev_procs.CmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, // srcStageMask
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // dstStageMask
            0, // dependencyFlags
            0, // memoryBarrierCount
            NULL, // pMemoryBarriers
            0, // bufferMemoryBarrierCount
            NULL, // pBufferMemoryBarriers
            1, // imageMemoryBarrierCount
            &color_image_barrier // pImageMemoryBarriers
        );
    }
    result = vk_dev_procs.EndCommandBuffer(command_buffer);
    assertVk(result);


    const VkPipelineStageFlags wait_dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkSubmitInfo submit_info {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &swapchain_image_acquired_semaphore,
        .pWaitDstStageMask = &wait_dst_stage_mask,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &this_frame_resources->render_finished_semaphore,
    };
    result = vk_dev_procs.QueueSubmit(queue_, 1, &submit_info, command_buffer_pending_fence);
    assertVk(result);


    VkPresentInfoKHR present_info {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &this_frame_resources->render_finished_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &p_surface_resources->swapchain,
        .pImageIndices = &acquired_swapchain_image_idx,
    };
    result = vk_dev_procs.QueuePresentKHR(queue_, &present_info);

    switch (result) {
        case VK_ERROR_OUT_OF_DATE_KHR: return RenderResult::error_surface_resources_out_of_date;
        case VK_SUBOPTIMAL_KHR: return RenderResult::success_surface_resources_out_of_date;
        default: assertVk(result);
    }

    return RenderResult::success;
}


extern VkInstance getVkInstance(void) {
    return instance_;
}

//
// ===========================================================================================================
//

} // namespace

