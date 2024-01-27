#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <dlfcn.h>

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

#include <shaderc/shaderc.h>

#include <tracy/tracy/Tracy.hpp>

#include "types.hpp"
#include "error_util.hpp"
#include "vk_procs.hpp"
#include "alloc_util.hpp"
#include "defer.hpp"
#include "math_util.hpp"
#include "graphics.hpp"
#include "libshaderc_procs.hpp"
#include "file_watch.hpp"

namespace graphics {

using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

//
// Type definitions that have to exist early on in the file :( very annoying =================================
//
// TODO move these to some header that you only include here, so you don't have to look at this shit at the
// top of this file. You can call that header `graphics_private.hpp` or something like that.

enum PipelineIndex {
    PIPELINE_INDEX_VOXEL_PIPELINE,
    PIPELINE_INDEX_GRID_PIPELINE,
    PIPELINE_INDEX_CUBE_OUTLINE_PIPELINE,

    PIPELINE_INDEX_COUNT,
};

using FN_CreatePipeline = bool (
    VkDevice device,
    VkShaderModule vertex_shader_module,
    VkShaderModule fragment_shader_module,
    VkRenderPass render_pass,
    u32 subpass,
    VkDescriptorSetLayout descriptor_set_layout,
    VkPipeline* pipeline_out,
    VkPipelineLayout* pipeline_layout_out
);
using PFN_CreatePipeline = FN_CreatePipeline*;

struct PipelineBuildFromSpirvFilesInfo {
    const char* vertex_shader_spirv_filepath;
    const char* fragment_shader_spirv_filepath;
    PFN_CreatePipeline pfn_createPipeline;
};

struct PipelineHotReloadInfo {
    const char* vertex_shader_src_filepath;
    const char* fragment_shader_src_filepath;
    PFN_CreatePipeline pfn_createPipeline;
};

struct PipelineAndLayout {
    VkPipeline pipeline;
    VkPipelineLayout layout;
};

struct ShaderSourceFileWatchIds {
    filewatch::FileID vertex_shader_id;
    filewatch::FileID fragment_shader_id;
};

struct GraphicsPipelineShaderModules {
    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;
};

//
// Forward declarations ======================================================================================
//

static FN_CreatePipeline createVoxelPipeline;
static FN_CreatePipeline createGridPipeline;
static FN_CreatePipeline createCubeOutlinePipeline;

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

const PipelineBuildFromSpirvFilesInfo PIPELINE_BUILD_FROM_SPIRV_FILES_INFOS[PIPELINE_INDEX_COUNT] {
    [PIPELINE_INDEX_VOXEL_PIPELINE] = {
        .vertex_shader_spirv_filepath = "build/voxel.vert.spv",
        .fragment_shader_spirv_filepath = "build/voxel.frag.spv",
        .pfn_createPipeline = createVoxelPipeline,
    },
    [PIPELINE_INDEX_GRID_PIPELINE] = {
        .vertex_shader_spirv_filepath = "build/grid.vert.spv",
        .fragment_shader_spirv_filepath = "build/grid.frag.spv",
        .pfn_createPipeline = createGridPipeline,
    },
    [PIPELINE_INDEX_CUBE_OUTLINE_PIPELINE] = {
        .vertex_shader_spirv_filepath = "build/cube_outline.vert.spv",
        .fragment_shader_spirv_filepath = "build/cube_outline.frag.spv",
        .pfn_createPipeline = createCubeOutlinePipeline,
    },
};

const PipelineHotReloadInfo PIPELINE_HOT_RELOAD_INFOS[PIPELINE_INDEX_COUNT] {
    [PIPELINE_INDEX_VOXEL_PIPELINE] = {
        .vertex_shader_src_filepath = "src/voxel.vert",
        .fragment_shader_src_filepath = "src/voxel.frag",
        .pfn_createPipeline = createVoxelPipeline,
    },
    [PIPELINE_INDEX_GRID_PIPELINE] = {
        .vertex_shader_src_filepath = "src/grid.vert",
        .fragment_shader_src_filepath = "src/grid.frag",
        .pfn_createPipeline = createGridPipeline,
    },
    [PIPELINE_INDEX_CUBE_OUTLINE_PIPELINE] = {
        .vertex_shader_src_filepath = "src/cube_outline.vert",
        .fragment_shader_src_filepath = "src/cube_outline.frag",
        .pfn_createPipeline = createCubeOutlinePipeline,
    },
};

//
// Global variables ==========================================================================================
//

static bool initialized_ = false;

static shaderc_compiler_t libshaderc_compiler_ = NULL;

static VkInstance instance_ = VK_NULL_HANDLE;
static VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
static VkPhysicalDeviceProperties physical_device_properties_ {};
static u32 queue_family_ = INVALID_QUEUE_FAMILY_IDX;
static VkDevice device_ = VK_NULL_HANDLE;
static VkQueue queue_ = VK_NULL_HANDLE;

static VkRenderPass simple_render_pass_ = VK_NULL_HANDLE;
static u32 the_only_subpass_ = INVALID_SUBPASS_IDX;

static PipelineAndLayout pipelines_[PIPELINE_INDEX_COUNT] {};
static GraphicsPipelineShaderModules shader_modules_[PIPELINE_INDEX_COUNT] {};

static VmaAllocator vma_allocator_ = NULL;

static VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;

static bool shader_source_file_watch_enabled_ = false;
static filewatch::Watchlist shader_source_file_watchlist_ = NULL;
static ShaderSourceFileWatchIds shader_source_file_watch_ids_[PIPELINE_INDEX_COUNT] {};

static bool grid_enabled_ = false;

//
// ===========================================================================================================
//

struct QueueFamilyRequirements {
    VkQueueFlags required_queue_flags;
    bool require_presentation_support;
};

struct PipelineInfo {
    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;
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

struct GridPipelineFragmentShaderPushConstants {
    alignas(16) mat4 world_to_screen_transform_inverse;
    alignas(16) vec2 viewport_offset_in_window;
    alignas( 8) vec2 viewport_size_in_window;
};

struct UniformBuffer {
    alignas(16) mat4 world_to_screen_transform;
};

struct RenderResourcesImpl {
    struct PerFrameResources {

        // OPTIMIZE:
        // Some of these resources are accessed at least once per frame.
        // Others are only accessed when attaching/detaching from a surface.
        // Maybe separate these into two arrays.

        // Lifetime: same as the lifetime of this RenderResourcesImpl.

        VkCommandBuffer command_buffer;
        VkFence command_buffer_pending_fence;
        // TODO FIXME: Do we need to destroy and recreate this when attaching/detaching from a swapchain,
        // just to be sure that it wasn't left signalled?
        VkSemaphore render_finished_semaphore;

        VkBuffer uniform_buffer;
        VmaAllocation uniform_buffer_allocation;
        VmaAllocationInfo uniform_buffer_allocation_info;

        VkBuffer voxels_buffer;
        VmaAllocation voxels_buffer_allocation;
        VmaAllocationInfo voxels_buffer_allocation_info;

        VkBuffer outlined_voxels_index_buffer;
        VmaAllocation outlined_voxels_index_buffer_allocation;
        VmaAllocationInfo outlined_voxels_index_buffer_allocation_info;

        VkDescriptorSet descriptor_set;

        // Lifetime: as long as this RenderResourcesImpl is attached to a SurfaceImpl.
        // In theory, we only need to destroy them when we attach to a surface of different size than these
        // resources; but we can destroy them for simplicity.

        VkFramebuffer framebuffer;

        VkImage render_target;
        VkImageView render_target_view;
        VmaAllocation render_target_allocation;

        VkImage depth_buffer;
        VkImageView depth_buffer_view;
        VmaAllocation depth_buffer_allocation;
    };

    VkRenderPass render_pass;
    VkCommandPool command_pool;

    VkBuffer voxels_staging_buffer;
    VmaAllocation voxels_staging_buffer_allocation;
    VmaAllocationInfo voxels_staging_buffer_allocation_info;

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
    VkSemaphore* swapchain_image_in_use_semaphores;

    VkExtent2D swapchain_extent;
    u32 swapchain_image_count;
    u32 last_used_swapchain_image_acquired_semaphore_idx;

    RenderResourcesImpl* attached_render_resources; // can be NULL if nothing is attached

    PresentModeFlags supported_present_modes;
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
    ZoneScoped;

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


/// You own the returned result. You are responsible for calling `shaderc_result_release()` to free it.
static shaderc_compilation_result_t compileShaderSrcFileToSpirv(
    const char* shader_src_filename,
    shaderc_shader_kind shader_type
) {
    ZoneScoped;

    // TODO: should we lock the source file while reading it? Maybe using something like `fcntl` or `flock`.

    size_t file_size = 0;
    void* file_contents = readEntireFile(shader_src_filename, &file_size);
    if (file_contents == NULL) {
        LOG_F(ERROR, "Failed to read shader src file `%s`.", shader_src_filename);
        return NULL;
    }
    defer(free(file_contents));

    shaderc_compilation_result_t compilation_result = libshaderc_procs_.compile_into_spv(
        libshaderc_compiler_,
        (const char*)file_contents, // source_text
        file_size, // source_text_size
        shader_type, // shader_kind
        shader_src_filename, // input_file_name
        "main", // entry_point_name
        0 // additional_options
    );
    alwaysAssert(compilation_result != NULL);

    return compilation_result;
}


[[nodiscard]] static VkResult createShaderModuleFromSpirv(
    VkDevice device,
    u32 spirv_byte_count,
    const u32* p_spirv_bytes,
    VkShaderModule* p_shader_module_out
) {
    ZoneScoped;

    VkShaderModuleCreateInfo cinfo {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv_byte_count,
        .pCode = (const u32*)p_spirv_bytes,
    };

    VkResult result = vk_dev_procs.CreateShaderModule(device, &cinfo, NULL, p_shader_module_out);
    return result;
};


/*
// TODO FIXME: return some indication of success or error
static void rebuildPipeline(
    VkShaderModule vertex_shader_module,
    VkShaderModule fragment_shader_module,
    PFN_CreatePipeline pfn_createPipeline,

    VkRenderPass render_pass,
    u32 subpass,
    VkDescriptorSetLayout descriptor_set_layout,

    VkPipeline* pipeline_in_out,
    VkPipelineLayout* pipeline_layout_in_out
) {
    // OPTIMIZE use pipeline cache? Maybe unnecessary complexity.

    // TODO FIXME: modify the create.*Pipeline procedures to return an error on failure whenever possible.

    VkPipeline new_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout new_pipeline_layout = VK_NULL_HANDLE;
    pfn_createPipeline(
        device_,
        vertex_shader_module, fragment_shader_module,
        render_pass, subpass, descriptor_set_layout,
        &new_pipeline, &new_pipeline_layout
    );
    // TODO FIXME: check for success here, return an error on failure

    vk_dev_procs.DestroyPipeline(device_, *pipeline_in_out, NULL);
    *pipeline_in_out = new_pipeline;

    // OPTIMIZE don't destroy and create pipeline_layout_in_out; it won't change
    vk_dev_procs.DestroyPipelineLayout(device_, *pipeline_layout_in_out, NULL);
    *pipeline_layout_in_out = new_pipeline_layout;
};
*/


/*
/// Caller is responsible for ensuring the pipeline and layout are not in use, e.g. via `vkDeviceWaitIdle`.
/// If `*pipeline_in_out` is not VK_NULL_HANDLE, destroys the old pipeline.
/// If `*pipeline_layout_in_out` is not VK_NULL_HANDLE, destroys the old pipeline layout.
static void hotReloadShadersAndPipeline(
    const char* vertex_shader_src_filename,
    const char* fragment_shader_src_filename,

    PFN_CreatePipeline pfn_createPipeline,

    VkRenderPass render_pass,
    u32 subpass,
    VkDescriptorSetLayout descriptor_set_layout,

    VkPipeline* pipeline_in_out,
    VkPipelineLayout* pipeline_layout_in_out
) {
    // OPTIMIZE only reload the shaders that have changed
    // OPTIMIZE don't destroy and create pipeline_layout_in_out; it won't change
    // OPTIMIZE use pipeline cache

    shaderc_compilation_result_t result_vertex = compileShaderSrcFileToSpirv(
        vertex_shader_src_filename,
        shaderc_glsl_vertex_shader
    );
    alwaysAssert(result_vertex != NULL);
    defer(libshaderc_procs_.result_release(result_vertex));

    size_t vertex_spirv_byte_count = libshaderc_procs_.result_get_length(result_vertex);
    const void* vertex_spirv_bytes = libshaderc_procs_.result_get_bytes(result_vertex);


    shaderc_compilation_result_t result_fragment = compileShaderSrcFileToSpirv(
        fragment_shader_src_filename,
        shaderc_glsl_fragment_shader
    );
    alwaysAssert(result_fragment != NULL);
    defer(libshaderc_procs_.result_release(result_fragment));

    size_t fragment_spirv_byte_count = libshaderc_procs_.result_get_length(result_fragment);
    const void* fragment_spirv_bytes = libshaderc_procs_.result_get_bytes(result_fragment);


    // TODO FIXME: Don't destroy these before confirming success of pfn_createPipeline.
    //     If it fails, just return an error or something, while keeping the old pipelines in place,
    //     so that the developer can fix the malformed shader or whatever without breaking things too much.
    //     This means we also need to modify the pipeline creation procedures to return an error where
    //     reasonable, instead of just asserting everywhere.
    if (pipeline_in_out != NULL) vk_dev_procs.DestroyPipeline(device_, *pipeline_in_out, NULL);
    if (pipeline_layout_in_out != NULL) vk_dev_procs.DestroyPipelineLayout(device_, *pipeline_layout_in_out, NULL);

    pfn_createPipeline(
        device_,
        (u32)vertex_spirv_byte_count,
        vertex_spirv_bytes,
        (u32)fragment_spirv_byte_count,
        fragment_spirv_bytes,
        render_pass,
        subpass,
        descriptor_set_layout,
        pipeline_in_out,
        pipeline_layout_in_out
    );
}
*/


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


[[nodiscard]] static bool createVoxelPipeline(
    VkDevice device,
    VkShaderModule vertex_shader_module,
    VkShaderModule fragment_shader_module,
    VkRenderPass render_pass,
    u32 subpass,
    VkDescriptorSetLayout descriptor_set_layout,
    VkPipeline* pipeline_out,
    VkPipelineLayout* pipeline_layout_out
) {

    VkResult result;


    VkSpecializationMapEntry vertex_shader_specialization_map_entry {
        .constantID = 0,
        .offset = 0,
        .size = sizeof(f32),
    };
    VkSpecializationInfo vertex_shader_specialization_info {
        .mapEntryCount = 1,
        .pMapEntries = &vertex_shader_specialization_map_entry,
        .dataSize = sizeof(f32),
        .pData = &VOXEL_RADIUS,
    };
    static_assert(sizeof(VOXEL_RADIUS) == sizeof(f32));

    constexpr u32 shader_stage_info_count = 2;
    const VkPipelineShaderStageCreateInfo shader_stage_infos[shader_stage_info_count] {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main",
            .pSpecializationInfo = &vertex_shader_specialization_info
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module,
            .pName = "main",
        },
    };


    VkVertexInputBindingDescription vertex_binding_description {
        .binding = 0,
        .stride = sizeof(Voxel),
        .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE,
    };

    constexpr u32 vertex_attribute_description_count = 2;
    VkVertexInputAttributeDescription vertex_attribute_descriptions[vertex_attribute_description_count] {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SINT,
            .offset = offsetof(Voxel, coord),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .offset = offsetof(Voxel, color),
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertex_input_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_binding_description,
        .vertexAttributeDescriptionCount = vertex_attribute_description_count,
        .pVertexAttributeDescriptions = vertex_attribute_descriptions,
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


    constexpr u32 push_constant_range_count = 0;
    VkPushConstantRange* push_constant_ranges = NULL;

    const VkPipelineLayoutCreateInfo pipeline_layout_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = push_constant_range_count,
        .pPushConstantRanges = push_constant_ranges,
    };

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    result = vk_dev_procs.CreatePipelineLayout(device, &pipeline_layout_info, NULL, &pipeline_layout);
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

    VkPipeline pipeline = VK_NULL_HANDLE;
    result = vk_dev_procs.CreateGraphicsPipelines(
        device,
        VK_NULL_HANDLE, // pipelineCache
        1, // createInfoCount
        &pipeline_info,
        NULL, // allocationCallbacks
        &pipeline
    );
    if (result == VK_ERROR_INVALID_SHADER_NV) {
        LOG_F(
            ERROR, "Failed to create pipeline for shader modules {%p, %p}.",
            vertex_shader_module, fragment_shader_module
        );
        vk_dev_procs.DestroyPipelineLayout(device, pipeline_layout, NULL);
        return false;
    }
    assertVk(result);


    *pipeline_layout_out = pipeline_layout;
    *pipeline_out = pipeline;

    return true;
}


[[nodiscard]] static bool createGridPipeline(
    VkDevice device,
    VkShaderModule vertex_shader_module,
    VkShaderModule fragment_shader_module,
    VkRenderPass render_pass,
    u32 subpass,
    VkDescriptorSetLayout descriptor_set_layout,
    VkPipeline* pipeline_out,
    VkPipelineLayout* pipeline_layout_out
) {

    VkResult result;


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
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = push_constant_range_count,
        .pPushConstantRanges = push_constant_ranges,
    };

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    result = vk_dev_procs.CreatePipelineLayout(device, &pipeline_layout_info, NULL, &pipeline_layout);
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

    VkPipeline pipeline = VK_NULL_HANDLE;
    result = vk_dev_procs.CreateGraphicsPipelines(
        device,
        VK_NULL_HANDLE, // pipelineCache
        1, // createInfoCount
        &pipeline_info,
        NULL, // allocationCallbacks
        &pipeline
    );
    if (result == VK_ERROR_INVALID_SHADER_NV) {
        LOG_F(
            ERROR, "Failed to create pipeline for shader modules {%p, %p}.",
            vertex_shader_module, fragment_shader_module
        );
        vk_dev_procs.DestroyPipelineLayout(device, pipeline_layout, NULL);
        return false;
    }
    assertVk(result);


    *pipeline_layout_out = pipeline_layout;
    *pipeline_out = pipeline;

    return true;
}


[[nodiscard]] static bool createCubeOutlinePipeline(
    VkDevice device,
    VkShaderModule vertex_shader_module,
    VkShaderModule fragment_shader_module,
    VkRenderPass render_pass,
    u32 subpass,
    VkDescriptorSetLayout descriptor_set_layout,
    VkPipeline* pipeline_out,
    VkPipelineLayout* pipeline_layout_out
) {

    VkResult result;


    VkSpecializationMapEntry vertex_shader_specialization_map_entry {
        .constantID = 0,
        .offset = 0,
        .size = sizeof(f32),
    };
    VkSpecializationInfo vertex_shader_specialization_info {
        .mapEntryCount = 1,
        .pMapEntries = &vertex_shader_specialization_map_entry,
        .dataSize = sizeof(f32),
        .pData = &VOXEL_RADIUS,
    };
    static_assert(sizeof(VOXEL_RADIUS) == sizeof(f32));

    constexpr u32 shader_stage_info_count = 2;
    const VkPipelineShaderStageCreateInfo shader_stage_infos[shader_stage_info_count] {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main",
            .pSpecializationInfo = &vertex_shader_specialization_info
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module,
            .pName = "main",
        },
    };


    VkVertexInputBindingDescription vertex_binding_description {
        .binding = 0,
        .stride = sizeof(u32),
        .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE,
    };
    VkVertexInputAttributeDescription vertex_attribute_description {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32_UINT,
        .offset = 0,
    };
    const VkPipelineVertexInputStateCreateInfo vertex_input_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_binding_description,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = &vertex_attribute_description,
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


    constexpr u32 push_constant_range_count = 0;
    VkPushConstantRange* push_constant_ranges = NULL;

    const VkPipelineLayoutCreateInfo pipeline_layout_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = push_constant_range_count,
        .pPushConstantRanges = push_constant_ranges,
    };

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    result = vk_dev_procs.CreatePipelineLayout(device, &pipeline_layout_info, NULL, &pipeline_layout);
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

    VkPipeline pipeline = VK_NULL_HANDLE;
    result = vk_dev_procs.CreateGraphicsPipelines(
        device,
        VK_NULL_HANDLE, // pipelineCache
        1, // createInfoCount
        &pipeline_info,
        NULL, // allocationCallbacks
        &pipeline
    );
    if (result == VK_ERROR_INVALID_SHADER_NV) {
        LOG_F(
            ERROR, "Failed to create pipeline for shader modules {%p, %p}.",
            vertex_shader_module, fragment_shader_module
        );
        vk_dev_procs.DestroyPipelineLayout(device, pipeline_layout, NULL);
        return false;
    }
    assertVk(result);


    *pipeline_layout_out = pipeline_layout;
    *pipeline_out = pipeline;

    return true;
}


/// On failure, sets `p_present_mode_count_out` to 0 and returns `NULL`.
/// You own the returned array. You are responsible for freeing it via `free()`.
static VkPresentModeKHR* getSupportedVkPresentModes(VkSurfaceKHR surface, u32* p_present_mode_count_out) {

    assert(initialized_);
    VkResult result;


    result = vk_inst_procs.GetPhysicalDeviceSurfacePresentModesKHR(
        physical_device_, surface, p_present_mode_count_out, NULL
    );
    assertVk(result);
    if (p_present_mode_count_out == 0) return NULL;

    VkPresentModeKHR* p_present_modes = mallocArray(*p_present_mode_count_out, VkPresentModeKHR);
    result = vk_inst_procs.GetPhysicalDeviceSurfacePresentModesKHR(
        physical_device_, surface, p_present_mode_count_out, p_present_modes
    );
    assertVk(result);


    return p_present_modes;
}


/// Doesn't verify present mode support. You must do so before calling this.
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

    #ifndef NDEBUG
    {
        u32 supported_mode_count = 0;
        VkPresentModeKHR* supported_modes = getSupportedVkPresentModes(surface, &supported_mode_count);

        bool present_mode_supported = false;
        for (u32fast i = 0; i < supported_mode_count; i++) {
            present_mode_supported |= (supported_modes[i] == present_mode);
        }
        assert(present_mode_supported);

        free(supported_modes);
    }
    #endif


    VkSurfaceCapabilitiesKHR surface_capabilities {};
    VkResult result = vk_inst_procs.GetPhysicalDeviceSurfaceCapabilitiesKHR(
        physical_device, surface, &surface_capabilities
    );
    assertVk(result);


    // Vk spec 1.3.259, appendix VK_KHR_swapchain, issue 12: suggests using capabilities.minImageCount + 1
    // to guarantee that vkAcquireNextImageKHR is non-blocking when using Mailbox present mode.
    // In FIFO mode, having a larger queue of submitted images increases input latency, so we request the
    // minimum possible in that case.
    // In Immediate mode, I assume the number of images doesn't matter.
    u32 min_image_count_request = surface_capabilities.minImageCount;
    if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) min_image_count_request += 1;
    {
        u32 count_preclamp = min_image_count_request;
        min_image_count_request = math::max(min_image_count_request, surface_capabilities.minImageCount);
        if (surface_capabilities.maxImageCount != 0) {
            min_image_count_request = math::min(min_image_count_request, surface_capabilities.maxImageCount);
        }

        if (min_image_count_request != count_preclamp) LOG_F(
            WARNING,
            "Min swapchain image count clamped from %" PRIu32 " to %" PRIu32 ", to fit surface limits.",
            count_preclamp, min_image_count_request
        );
    }


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


    LOG_F(
        INFO, "Requesting minImageCount=%" PRIu32 ", presentMode=%i for swapchain creation.",
        min_image_count_request, present_mode
    );
    VkSwapchainCreateInfoKHR swapchain_info {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = min_image_count_request,
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


static void createPerSwapchainImageSurfaceResources(
    VkSwapchainKHR swapchain,
    u32* image_count_out,
    VkImage** images_out,
    VkSemaphore** image_acquired_semaphores_out,
    VkSemaphore** image_in_use_semaphores_out
) {

    u32 image_count = 0;
    VkResult result = vk_dev_procs.GetSwapchainImagesKHR(device_, swapchain, &image_count, NULL);
    assertVk(result);

    LOG_F(INFO, "Got %" PRIu32 " images from swapchain %p.", image_count, swapchain);


    VkImage* swapchain_images = mallocArray(image_count, VkImage);
    result = vk_dev_procs.GetSwapchainImagesKHR(device_, swapchain, &image_count, swapchain_images);
    assertVk(result);


    VkSemaphore* swapchain_image_acquired_semaphores = mallocArray(image_count, VkSemaphore);
    {
        VkSemaphoreCreateInfo semaphore_info { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        for (u32 im_idx = 0; im_idx < image_count; im_idx++) {
            result = vk_dev_procs.CreateSemaphore(
                device_, &semaphore_info, NULL, &swapchain_image_acquired_semaphores[im_idx]
            );
            assertVk(result);
        }
    }

    VkSemaphore* swapchain_image_in_use_semaphores = mallocArray(image_count, VkSemaphore);
    {
        VkSemaphoreCreateInfo semaphore_info { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        for (u32 im_idx = 0; im_idx < image_count; im_idx++) {
            result = vk_dev_procs.CreateSemaphore(
                device_, &semaphore_info, NULL, &swapchain_image_in_use_semaphores[im_idx]
            );
            assertVk(result);
        }
    }

    // set the image_in_use semaphores to signalled, so that we don't deadlock on the first frame
    {
        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        result = vk_dev_procs.CreateFence(device_, &fence_info, NULL, &fence);
        assertVk(result);

        VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 0,
            .pCommandBuffers = NULL,
            .signalSemaphoreCount = image_count,
            .pSignalSemaphores = swapchain_image_in_use_semaphores,
        };
        result = vk_dev_procs.QueueSubmit(queue_, 1, &submit_info, fence);
        assertVk(result);

        result = vk_dev_procs.WaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
        assertVk(result);
        vk_dev_procs.DestroyFence(device_, fence, NULL);
    }


    *image_count_out = image_count;
    *images_out = swapchain_images;
    *image_acquired_semaphores_out = swapchain_image_acquired_semaphores;
    *image_in_use_semaphores_out = swapchain_image_in_use_semaphores;
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
    const RenderResourcesImpl::PerFrameResources* p_frame_resources,
    u32 voxel_count,
    u32 outlined_voxel_count,
    VkRenderPass render_pass,
    VkExtent2D swapchain_extent,
    VkRect2D swapchain_roi,
    const GridPipelineFragmentShaderPushConstants* grid_pipeline_push_constants,
    ImDrawData* imgui_draw_data
) {
    ZoneScoped;

    VkCommandBuffer command_buffer = p_frame_resources->command_buffer;

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
        .framebuffer = p_frame_resources->framebuffer,
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


    {
        PipelineAndLayout* p_pipeline = &pipelines_[PIPELINE_INDEX_VOXEL_PIPELINE];

        vk_dev_procs.CmdBindDescriptorSets(
            command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, p_pipeline->layout,
            0, // firstSet
            1, // descriptorSetCount
            &p_frame_resources->descriptor_set,
            0, // dynamicOffsetCount
            NULL // pDynamicOffsets
        );

        vk_dev_procs.CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, p_pipeline->pipeline);

        VkDeviceSize offset_in_voxels_vertex_buf = 0;
        vk_dev_procs.CmdBindVertexBuffers(
            command_buffer, 0, 1, &p_frame_resources->voxels_buffer, &offset_in_voxels_vertex_buf
        );

        vk_dev_procs.CmdDraw(command_buffer, 36, voxel_count, 0, 0);
    }
    {
        PipelineAndLayout* p_pipeline = &pipelines_[PIPELINE_INDEX_CUBE_OUTLINE_PIPELINE];

        vk_dev_procs.CmdBindDescriptorSets(
            command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, p_pipeline->layout,
            0, // firstSet
            1, // descriptorSetCount
            &p_frame_resources->descriptor_set,
            0, // dynamicOffsetCount
            NULL // pDynamicOffsets
        );

        vk_dev_procs.CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, p_pipeline->pipeline);

        VkDeviceSize offset_in_outlined_voxels_index_buf = 0;
        vk_dev_procs.CmdBindVertexBuffers(
            command_buffer, 0, 1, &p_frame_resources->outlined_voxels_index_buffer, &offset_in_outlined_voxels_index_buf
        );

        vk_dev_procs.CmdDraw(command_buffer, 72, outlined_voxel_count, 0, 0);
    }
    if (grid_enabled_) {
        PipelineAndLayout* p_pipeline = &pipelines_[PIPELINE_INDEX_GRID_PIPELINE];

        vk_dev_procs.CmdBindDescriptorSets(
            command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, p_pipeline->layout,
            0, // firstSet
            1, // descriptorSetCount
            &p_frame_resources->descriptor_set,
            0, // dynamicOffsetCount
            NULL // pDynamicOffsets
        );
        vk_dev_procs.CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, p_pipeline->pipeline);

        vk_dev_procs.CmdPushConstants(
            command_buffer, p_pipeline->layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
            sizeof(*grid_pipeline_push_constants), grid_pipeline_push_constants
        );

        vk_dev_procs.CmdDraw(command_buffer, 6, 1, 0, 0);
    }


    if (imgui_draw_data != NULL) {
        ZoneScopedN("ImGui_ImplVulkan_RenderDrawData");
        ImGui_ImplVulkan_RenderDrawData(imgui_draw_data, command_buffer);
    }

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

    ZoneScoped;

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


    constexpr u32 descriptor_set_layout_binding_count = 2;
    VkDescriptorSetLayoutBinding descriptor_set_layout_bindings[descriptor_set_layout_binding_count] {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = NULL,
        },
        // voxels
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = NULL,
        },
    };
    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = descriptor_set_layout_binding_count,
        .pBindings = descriptor_set_layout_bindings,
    };
    result = vk_dev_procs.CreateDescriptorSetLayout(
        device_, &descriptor_set_layout_info, NULL, &descriptor_set_layout_
    );
    assertVk(result);


    the_only_subpass_ = 0;

    {
        ZoneScopedN("pipeline init");

        for (u32fast pipeline_idx = 0; pipeline_idx < PIPELINE_INDEX_COUNT; pipeline_idx++) {

            const PipelineBuildFromSpirvFilesInfo* p_build_info =
                &PIPELINE_BUILD_FROM_SPIRV_FILES_INFOS[pipeline_idx];


            size_t vertex_shader_spirv_byte_count = 0;
            void* vertex_shader_spirv_bytes = readEntireFile(
                p_build_info->vertex_shader_spirv_filepath,
                &vertex_shader_spirv_byte_count
            );
            alwaysAssert(vertex_shader_spirv_bytes != NULL);
            defer(free(vertex_shader_spirv_bytes));

            alwaysAssert(vertex_shader_spirv_byte_count % sizeof(u32) == 0);
            alwaysAssert((uintptr_t)vertex_shader_spirv_bytes % alignof(u32) == 0);

            VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
            result = createShaderModuleFromSpirv(
                device_,
                (u32)vertex_shader_spirv_byte_count,
                (const u32*)vertex_shader_spirv_bytes,
                &vertex_shader_module
            );
            assertVk(result);
            shader_modules_[pipeline_idx].vertex_shader_module = vertex_shader_module;


            size_t fragment_shader_spirv_byte_count = 0;
            void* fragment_shader_spirv_bytes = readEntireFile(
                p_build_info->fragment_shader_spirv_filepath,
                &fragment_shader_spirv_byte_count
            );
            alwaysAssert(fragment_shader_spirv_bytes != NULL);
            defer(free(fragment_shader_spirv_bytes));

            alwaysAssert(vertex_shader_spirv_byte_count % sizeof(u32) == 0);
            alwaysAssert((uintptr_t)vertex_shader_spirv_bytes % alignof(u32) == 0);

            VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
            result = createShaderModuleFromSpirv(
                device_,
                (u32)fragment_shader_spirv_byte_count,
                (const u32*)fragment_shader_spirv_bytes,
                &fragment_shader_module
            );
            assertVk(result);
            shader_modules_[pipeline_idx].fragment_shader_module = fragment_shader_module;


            PipelineAndLayout* p_pipeline = &pipelines_[pipeline_idx];
            bool success = p_build_info->pfn_createPipeline(
                device_,
                vertex_shader_module, fragment_shader_module,
                render_pass, the_only_subpass_, descriptor_set_layout_,
                &p_pipeline->pipeline, &p_pipeline->layout
            );
            alwaysAssert(success);
        }
    }


    {
        ZoneScopedN("libshaderc init");
        bool success = libshaderc_procs_.init();
        // TODO FIXME we should just log an error message, disable the hot-reloading feature, and continue
        // running. This isn't fatal, after all.
        alwaysAssert(success);

        libshaderc_compiler_ = libshaderc_procs_.compiler_initialize();
        alwaysAssert(libshaderc_compiler_ != NULL);
    }


    initialized_ = true;
}


extern PresentModeFlags getSupportedPresentModes(SurfaceResources surface_resources) {
    const SurfaceResourcesImpl* p_surface_resources = (const SurfaceResourcesImpl*)surface_resources.impl;
    return p_surface_resources->supported_present_modes;
};


/// Returns VK_PRESENT_MODE_MAX_ENUM_KHR if there is no mode in `p_modes` whose priority > 0.
static VkPresentModeKHR selectHighestPriorityPresentMode(
    const PresentModePriorities priorities,
    u32 mode_count,
    VkPresentModeKHR* p_modes
) {
    VkPresentModeKHR highest_priority_mode = VK_PRESENT_MODE_MAX_ENUM_KHR;
    u8 highest_priority = 0;

    for (u32fast mode_idx = 0; mode_idx < mode_count; mode_idx++) {

        VkPresentModeKHR mode = p_modes[mode_idx];
        if (mode >= (int)PRESENT_MODE_ENUM_COUNT) continue;

        u8 priority = priorities[mode];
        if (priority <= highest_priority) continue;

        highest_priority_mode = mode;
        highest_priority = priority;
    }

    return highest_priority_mode;
};


extern Result createSurfaceResources(
    VkSurfaceKHR surface,
    const PresentModePriorities present_mode_priorities,
    VkExtent2D fallback_window_size,
    SurfaceResources* surface_resources_out,
    PresentMode* selected_present_mode_out
) {
    ZoneScoped;

    SurfaceResourcesImpl* p_resources = (SurfaceResourcesImpl*)calloc(1, sizeof(SurfaceResourcesImpl));
    assertErrno(p_resources != NULL);


    p_resources->surface = surface;
    p_resources->attached_render_resources = NULL;
    p_resources->last_used_swapchain_image_acquired_semaphore_idx = 0;


    VkPresentModeKHR present_mode;
    {
        u32 supported_mode_count = 0;
        VkPresentModeKHR* p_supported_modes = getSupportedVkPresentModes(surface, &supported_mode_count);
        alwaysAssert(p_supported_modes != NULL);
        defer(free(p_supported_modes));

        PresentModeFlags supported_mode_flags = 0;
        for (u32fast i = 0; i < supported_mode_count; i++) {
            VkPresentModeKHR mode = p_supported_modes[i];
            if (p_supported_modes[i] < (int)PRESENT_MODE_ENUM_COUNT) {
                supported_mode_flags |= PresentModeFlagBits_fromMode((PresentMode)mode);
            }
        }

        p_resources->supported_present_modes = supported_mode_flags;

        present_mode = selectHighestPriorityPresentMode(
            present_mode_priorities, supported_mode_count, p_supported_modes
        );
        alwaysAssert(present_mode != VK_PRESENT_MODE_MAX_ENUM_KHR);
    }


    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D swapchain_extent {};
    Result res = createSwapchain(
        physical_device_, device_, surface, fallback_window_size, queue_family_, present_mode,
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
        &p_resources->swapchain_image_acquired_semaphores,
        &p_resources->swapchain_image_in_use_semaphores
    );


    surface_resources_out->impl = p_resources;
    if (selected_present_mode_out != NULL) *selected_present_mode_out = (PresentMode)present_mode;
    return Result::success;
}


/// `renderer` must not currently have any surface attached.
/// `surface` must not currently have any renderer attached.
extern void attachSurfaceToRenderer(SurfaceResources surface, RenderResources renderer) {
    ZoneScoped;

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
    ZoneScoped;

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


/// Use if a window resize has caused the resources to be out-of-date, or to switch present modes.
extern Result updateSurfaceResources(
    SurfaceResources surface_resources,
    const PresentModePriorities present_mode_priorities,
    VkExtent2D fallback_window_size,
    PresentMode* selected_present_mode_out
) {
    ZoneScoped;

    SurfaceResourcesImpl* p_surface_resources = (SurfaceResourcesImpl*)surface_resources.impl;


    VkPresentModeKHR present_mode;
    {
        u32 supported_mode_count = 0;
        VkPresentModeKHR* p_supported_modes = getSupportedVkPresentModes(
            p_surface_resources->surface, &supported_mode_count
        );
        alwaysAssert(p_supported_modes != NULL);
        defer(free(p_supported_modes));

        PresentModeFlags supported_mode_flags = 0;
        for (u32fast i = 0; i < supported_mode_count; i++) {
            VkPresentModeKHR mode = p_supported_modes[i];
            if (p_supported_modes[i] < (int)PRESENT_MODE_ENUM_COUNT) {
                supported_mode_flags |= PresentModeFlagBits_fromMode((PresentMode)mode);
            }
        }

        p_surface_resources->supported_present_modes = supported_mode_flags;

        present_mode = selectHighestPriorityPresentMode(
            present_mode_priorities, supported_mode_count, p_supported_modes
        );
        alwaysAssert(present_mode != VK_PRESENT_MODE_MAX_ENUM_KHR);
    }


    VkSwapchainKHR old_swapchain = p_surface_resources->swapchain;
    VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;

    Result res = createSwapchain(
        physical_device_, device_, p_surface_resources->surface, fallback_window_size, queue_family_,
        present_mode, old_swapchain, &new_swapchain, &p_surface_resources->swapchain_extent
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
            &p_surface_resources->swapchain_image_acquired_semaphores,
            &p_surface_resources->swapchain_image_in_use_semaphores
        );

        p_surface_resources->last_used_swapchain_image_acquired_semaphore_idx = 0;
    }

    RenderResourcesImpl* p_render_resources = p_surface_resources->attached_render_resources;
    if (p_render_resources != NULL) { // a renderer is attached
        // OPTIMIZE: defaulted to nuclear option here; consider whether we want to do this more efficiently.
        detachSurfaceFromRenderer(surface_resources, RenderResources { .impl = p_render_resources });
        attachSurfaceToRenderer(surface_resources, RenderResources { .impl = p_render_resources });
    }

    if (selected_present_mode_out != NULL) *selected_present_mode_out = (PresentMode)present_mode;
    return Result::success;
}


extern Result createRenderer(RenderResources* render_resources_out) {
    ZoneScoped;

    VkResult result;

    RenderResourcesImpl* p_render_resources = (RenderResourcesImpl*)calloc(1, sizeof(RenderResourcesImpl));
    assertErrno(p_render_resources != NULL);


    p_render_resources->render_pass = simple_render_pass_;


    constexpr u32 descriptor_pool_size_count = 2;
    VkDescriptorPoolSize descriptor_pool_sizes[descriptor_pool_size_count] {
        {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT,
        },
        {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT,
        },
    };
    VkDescriptorPoolCreateInfo descriptor_pool_info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = MAX_FRAMES_IN_FLIGHT,
        .poolSizeCount = descriptor_pool_size_count,
        .pPoolSizes = descriptor_pool_sizes,
    };

    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    result = vk_dev_procs.CreateDescriptorPool(device_, &descriptor_pool_info, NULL, &descriptor_pool);
    assertVk(result);

    VkDescriptorSetLayout descriptor_set_layouts[MAX_FRAMES_IN_FLIGHT];
    for (u32fast i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        descriptor_set_layouts[i] = descriptor_set_layout_;
    VkDescriptorSetAllocateInfo descriptor_set_alloc_info {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
        .pSetLayouts = descriptor_set_layouts,
    };

    VkDescriptorSet descriptor_sets[MAX_FRAMES_IN_FLIGHT] {};
    result = vk_dev_procs.AllocateDescriptorSets(device_, &descriptor_set_alloc_info, descriptor_sets);
    assertVk(result);
    // NOTE: these descriptor sets need to be copied into the per-frame structs in this procedure


    const VkCommandPoolCreateInfo command_pool_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family_,
    };

    result = vk_dev_procs.CreateCommandPool(
        device_, &command_pool_info, NULL, &p_render_resources->command_pool
    );
    assertVk(result);

    VkCommandBuffer command_buffers[MAX_FRAMES_IN_FLIGHT] {};
    {
        VkCommandBufferAllocateInfo cmd_buf_alloc_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = p_render_resources->command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
        };
        result = vk_dev_procs.AllocateCommandBuffers(device_, &cmd_buf_alloc_info, command_buffers);
        assertVk(result);
    }
    // NOTE: these command buffers need to be copied into the per-frame structs in this procedure


    for (u32fast frame_idx = 0; frame_idx < MAX_FRAMES_IN_FLIGHT; frame_idx++) {

        RenderResourcesImpl::PerFrameResources* this_frame_resources =
            &p_render_resources->frame_resources_array[frame_idx];

        this_frame_resources->command_buffer = command_buffers[frame_idx];
        this_frame_resources->descriptor_set = descriptor_sets[frame_idx];

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

        VkBuffer* p_uniform_buffer = &this_frame_resources->uniform_buffer;
        VmaAllocation* p_uniform_buffer_allocation = &this_frame_resources->uniform_buffer_allocation;
        VmaAllocationInfo* p_uniform_buffer_allocation_info = &this_frame_resources->uniform_buffer_allocation_info;
        {
            // TODO are we guaranteed to have a memory type supporting both HOST_VISIBLE and USAGE_UNIFORM_BUFFER?
            VkBufferCreateInfo uniform_buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = sizeof(UniformBuffer),
                .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 1,
                .pQueueFamilyIndices = &queue_family_,
            };
            VmaAllocationCreateInfo uniform_buffer_alloc_info {
                .usage = VMA_MEMORY_USAGE_AUTO,
                .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            };
            result = vmaCreateBuffer(
                vma_allocator_, &uniform_buffer_info, &uniform_buffer_alloc_info,
                p_uniform_buffer, p_uniform_buffer_allocation, p_uniform_buffer_allocation_info
            );
            assertVk(result);
        }

        VkBuffer* p_voxels_buffer = &this_frame_resources->voxels_buffer;
        VmaAllocation* p_voxels_buffer_allocation = &this_frame_resources->voxels_buffer_allocation;
        VmaAllocationInfo* p_voxels_buffer_allocation_info = &this_frame_resources->voxels_buffer_allocation_info;
        {
            // TODO are we guaranteed to have a memory type supporting both HOST_VISIBLE and USAGE_VERTEX_BUFFER?
            VkBufferCreateInfo voxels_buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = MAX_VOXEL_COUNT * sizeof(Voxel),
                .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 1,
                .pQueueFamilyIndices = &queue_family_,
            };
            VmaAllocationCreateInfo voxels_buffer_alloc_info {
                .usage = VMA_MEMORY_USAGE_AUTO,
                .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            };
            result = vmaCreateBuffer(
                vma_allocator_, &voxels_buffer_info, &voxels_buffer_alloc_info,
                p_voxels_buffer, p_voxels_buffer_allocation, p_voxels_buffer_allocation_info
            );
            assertVk(result);
        }

        VkBuffer* p_outlined_voxels_index_buffer = &this_frame_resources->outlined_voxels_index_buffer;
        VmaAllocation* p_outlined_voxels_index_buffer_allocation = &this_frame_resources->outlined_voxels_index_buffer_allocation;
        VmaAllocationInfo* p_outlined_voxels_index_buffer_allocation_info = &this_frame_resources->outlined_voxels_index_buffer_allocation_info;
        {
            // TODO are we guaranteed to have a memory type supporting both HOST_VISIBLE and USAGE_VERTEX_BUFFER?
            VkBufferCreateInfo cube_outlines_index_buffer_info {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = MAX_OUTLINED_VOXEL_COUNT * sizeof(u32),
                .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 1,
                .pQueueFamilyIndices = &queue_family_,
            };
            VmaAllocationCreateInfo cube_outlines_index_buffer_alloc_info {
                .usage = VMA_MEMORY_USAGE_AUTO,
                .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            };
            result = vmaCreateBuffer(
                vma_allocator_, &cube_outlines_index_buffer_info, &cube_outlines_index_buffer_alloc_info,
                p_outlined_voxels_index_buffer, p_outlined_voxels_index_buffer_allocation, p_outlined_voxels_index_buffer_allocation_info
            );
            assertVk(result);
        }
    }


    constexpr u32 descriptors_per_frame_count = 2;
    constexpr u32 descriptor_write_count = MAX_FRAMES_IN_FLIGHT * descriptors_per_frame_count;

    VkDescriptorBufferInfo descriptor_buffer_infos[descriptor_write_count] {};
    VkWriteDescriptorSet descriptor_writes[descriptor_write_count] {};
    u32fast descriptor_write_idx = 0;

    for (u32fast frame_idx = 0; frame_idx < MAX_FRAMES_IN_FLIGHT; frame_idx++) {

        {
            descriptor_buffer_infos[descriptor_write_idx] = VkDescriptorBufferInfo {
                .buffer = p_render_resources->frame_resources_array[frame_idx].uniform_buffer,
                .offset = 0,
                .range = sizeof(UniformBuffer),
            };
            descriptor_writes[descriptor_write_idx] = VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = p_render_resources->frame_resources_array[frame_idx].descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pImageInfo = NULL,
                .pBufferInfo = &descriptor_buffer_infos[descriptor_write_idx],
                .pTexelBufferView = NULL,
            };
        }
        descriptor_write_idx++;

        {
            descriptor_buffer_infos[descriptor_write_idx] = VkDescriptorBufferInfo {
                .buffer = p_render_resources->frame_resources_array[frame_idx].voxels_buffer,
                .offset = 0,
                .range = MAX_VOXEL_COUNT * sizeof(Voxel),
            };
            descriptor_writes[descriptor_write_idx] = VkWriteDescriptorSet {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = p_render_resources->frame_resources_array[frame_idx].descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = NULL,
                .pBufferInfo = &descriptor_buffer_infos[descriptor_write_idx],
                .pTexelBufferView = NULL,
            };
        }
        descriptor_write_idx++;
    }
    vk_dev_procs.UpdateDescriptorSets(
         device_,
         descriptor_write_count, descriptor_writes,
         0, NULL
    );


    p_render_resources->last_used_frame_idx = 0;


    render_resources_out->impl = p_render_resources;
    return Result::success;
}


/// Returns whether it succeeded..
/// On failure, leaves `p_shader_module_out` unmodified.
[[nodiscard]] static bool createShaderModuleFromShaderSourceFile(
    VkDevice device,
    const char* shader_src_filepath,
    shaderc_shader_kind shader_type,
    VkShaderModule* p_shader_module_out
) {
    ZoneScoped;

    shaderc_compilation_result_t compile_result = compileShaderSrcFileToSpirv(
        shader_src_filepath, shader_type
    );
    if (compile_result == NULL) {
        LOG_F(ERROR, "Failed to compile shader src file `%s` to spirv.", shader_src_filepath);
        return false;
    };
    defer(libshaderc_procs_.result_release(compile_result));

    shaderc_compilation_status compile_status = libshaderc_procs_.result_get_compilation_status(compile_result);
    if (compile_status != shaderc_compilation_status_success) {

        u32fast error_count = libshaderc_procs_.result_get_num_errors(compile_result);
        const char* error_message = libshaderc_procs_.result_get_error_message(compile_result);
        if (error_message == NULL) error_message = "(NO ERROR MESSAGE PROVIDED)";

        LOG_F(
             ERROR, "Failed to compile shader `%s`. %" PRIuFAST32 " errors: `%s`.",
             shader_src_filepath, error_count, error_message
         );
        return false;
    }

    size_t spirv_byte_count = libshaderc_procs_.result_get_length(compile_result);
    const void* p_spirv_bytes = (const void*)libshaderc_procs_.result_get_bytes(compile_result);

    alwaysAssert(spirv_byte_count % sizeof(u32) == 0);
    alwaysAssert((uintptr_t)p_spirv_bytes % alignof(u32) == 0);

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkResult result = createShaderModuleFromSpirv(
        device, (u32)spirv_byte_count, (const u32*)p_spirv_bytes, &shader_module
    );
    if (result == VK_ERROR_INVALID_SHADER_NV) {
        LOG_F(
            ERROR, "Failed to create shader module from spirv for shader `%s`: VK_ERROR_INVALID_SHADER_NV.",
            shader_src_filepath
        );
        return false;
    }
    assertVk(result);

    *p_shader_module_out = shader_module;
    return true;
}


bool reloadAllShaders(RenderResources renderer) {

    ZoneScoped;

    LOG_F(INFO, "Reloading all shaders");


    timespec start_time {};
    {
        int success = timespec_get(&start_time, TIME_UTC);
        LOG_IF_F(ERROR, !success, "Failed to get shader rebuild start time.");
    }


    const RenderResourcesImpl* p_render_resources = (const RenderResourcesImpl*)renderer.impl;
    assert(p_render_resources != NULL);


    GraphicsPipelineShaderModules new_shader_modules[PIPELINE_INDEX_COUNT];
    memcpy(new_shader_modules, shader_modules_, PIPELINE_INDEX_COUNT * sizeof(GraphicsPipelineShaderModules));

    for (u32fast pipeline_idx = 0; pipeline_idx < PIPELINE_INDEX_COUNT; pipeline_idx++) {

        bool success;

        VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
        success = createShaderModuleFromShaderSourceFile(
            device_,
            PIPELINE_HOT_RELOAD_INFOS[pipeline_idx].vertex_shader_src_filepath,
            shaderc_glsl_vertex_shader,
            &vertex_shader_module
        );
        if (!success) return false;
        new_shader_modules[pipeline_idx].vertex_shader_module = vertex_shader_module;

        VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
        success = createShaderModuleFromShaderSourceFile(
            device_,
            PIPELINE_HOT_RELOAD_INFOS[pipeline_idx].fragment_shader_src_filepath,
            shaderc_glsl_fragment_shader,
            &fragment_shader_module
        );
        if (!success) return false;
        new_shader_modules[pipeline_idx].fragment_shader_module = fragment_shader_module;
    }


    PipelineAndLayout new_pipelines[PIPELINE_INDEX_COUNT];
    memcpy(new_pipelines, pipelines_, PIPELINE_INDEX_COUNT * sizeof(PipelineAndLayout));

    for (u32fast pipeline_idx = 0; pipeline_idx < PIPELINE_INDEX_COUNT; pipeline_idx++) {

        bool success = PIPELINE_HOT_RELOAD_INFOS[pipeline_idx].pfn_createPipeline(
            device_,
            new_shader_modules[pipeline_idx].vertex_shader_module,
            new_shader_modules[pipeline_idx].fragment_shader_module,
            p_render_resources->render_pass,
            the_only_subpass_,
            descriptor_set_layout_,
            &new_pipelines[pipeline_idx].pipeline,
            &new_pipelines[pipeline_idx].layout
        );
        if (!success) return false;
    }


    VkResult result = vk_dev_procs.QueueWaitIdle(queue_);
    assertVk(result);

    for (u32fast pipeline_idx = 0; pipeline_idx < PIPELINE_INDEX_COUNT; pipeline_idx++) {
        vk_dev_procs.DestroyShaderModule(device_, shader_modules_[pipeline_idx].vertex_shader_module, NULL);
        vk_dev_procs.DestroyShaderModule(device_, shader_modules_[pipeline_idx].fragment_shader_module, NULL);
    }
    memcpy(shader_modules_, new_shader_modules, sizeof(shader_modules_));

    for (u32fast pipeline_idx = 0; pipeline_idx < PIPELINE_INDEX_COUNT; pipeline_idx++) {
        vk_dev_procs.DestroyPipeline(device_, pipelines_[pipeline_idx].pipeline, NULL);
        // OPTIMIZE we probably don't need to rebuild the layout
        vk_dev_procs.DestroyPipelineLayout(device_, pipelines_[pipeline_idx].layout, NULL);
    }
    memcpy(pipelines_, new_pipelines, sizeof(pipelines_));


    timespec end_time;
    {
        int success = timespec_get(&end_time, TIME_UTC);
        LOG_IF_F(ERROR, !success, "Failed to get shader rebuild end time.");
    }

    f64 duration_milliseconds =
        (f64)(end_time.tv_sec - start_time.tv_sec) * 1'000. +
        (f64)(end_time.tv_nsec - start_time.tv_nsec) / 1'000'000.;
    LOG_F(INFO, "Shaders reloaded (%.0lf ms).", duration_milliseconds);


    return true;
}


RenderResult render(
    SurfaceResources surface,
    VkRect2D window_subregion,
    const mat4* world_to_screen_transform,
    const mat4* world_to_screen_transform_inverse,
    ImDrawData* imgui_draw_data,
    u32 voxel_count,
    const Voxel* p_voxels,
    u32 outlined_voxel_index_count,
    const u32* p_outlined_voxel_indices
) {

    ZoneScoped;

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

        {
            ZoneScopedN("acquireNextImage");
            result = vk_dev_procs.AcquireNextImageKHR(
                device_,
                p_surface_resources->swapchain,
                UINT64_MAX,
                swapchain_image_acquired_semaphore,
                VK_NULL_HANDLE,
                &acquired_swapchain_image_idx
            );
        }

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

    {
        ZoneScopedN("wait for command_buffer_pending_fence");
        result = vk_dev_procs.WaitForFences(device_, 1, &command_buffer_pending_fence, VK_TRUE, UINT64_MAX);
        assertVk(result);
    }

    result = vk_dev_procs.ResetFences(device_, 1, &command_buffer_pending_fence);
    assertVk(result);


    // upload data
    {
        // OPTIMIZE keep stuff persistently mapped?
        {
            VkMappedMemoryRange mapped_memory_range {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = this_frame_resources->uniform_buffer_allocation_info.deviceMemory,
                .offset = this_frame_resources->uniform_buffer_allocation_info.offset,
                .size = this_frame_resources->uniform_buffer_allocation_info.size,
            };

            void* ptr_to_mapped_memory = NULL;
            result = vk_dev_procs.MapMemory(
                device_,
                mapped_memory_range.memory,
                mapped_memory_range.offset,
                mapped_memory_range.size,
                0, // flags
                &ptr_to_mapped_memory
            );
            assertVk(result);

            UniformBuffer uniform_data {
                // OPTIMIZE copying 64 bytes here, is that a lot?
                .world_to_screen_transform = *world_to_screen_transform
            };
            *(UniformBuffer*)ptr_to_mapped_memory = uniform_data;

            result = vk_dev_procs.FlushMappedMemoryRanges(device_, 1, &mapped_memory_range);
            assertVk(result);

            vk_dev_procs.UnmapMemory(device_, mapped_memory_range.memory);
        }

        {
            VkMappedMemoryRange mapped_memory_range {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = this_frame_resources->voxels_buffer_allocation_info.deviceMemory,
                .offset = this_frame_resources->voxels_buffer_allocation_info.offset,
                .size = glm::ceilMultiple(
                    voxel_count * sizeof(Voxel),
                    physical_device_properties_.limits.nonCoherentAtomSize
                ),
            };

            void* ptr_to_mapped_memory = NULL;
            result = vk_dev_procs.MapMemory(
                device_,
                mapped_memory_range.memory,
                mapped_memory_range.offset,
                mapped_memory_range.size,
                0, // flags
                &ptr_to_mapped_memory
            );

            memcpy(ptr_to_mapped_memory, p_voxels, voxel_count * sizeof(Voxel));

            result = vk_dev_procs.FlushMappedMemoryRanges(device_, 1, &mapped_memory_range);
            assertVk(result);

            vk_dev_procs.UnmapMemory(device_, mapped_memory_range.memory);
        }

        if (outlined_voxel_index_count != 0) {
            VkMappedMemoryRange mapped_memory_range {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = this_frame_resources->outlined_voxels_index_buffer_allocation_info.deviceMemory,
                .offset = this_frame_resources->outlined_voxels_index_buffer_allocation_info.offset,
                .size = glm::ceilMultiple(
                    outlined_voxel_index_count * sizeof(u32),
                    physical_device_properties_.limits.nonCoherentAtomSize
                ),
            };

            void* ptr_to_mapped_memory = NULL;
            result = vk_dev_procs.MapMemory(
                device_,
                mapped_memory_range.memory,
                mapped_memory_range.offset,
                mapped_memory_range.size,
                0, // flags
                &ptr_to_mapped_memory
            );

            memcpy(ptr_to_mapped_memory, p_outlined_voxel_indices, outlined_voxel_index_count * sizeof(u32));

            result = vk_dev_procs.FlushMappedMemoryRanges(device_, 1, &mapped_memory_range);
            assertVk(result);

            vk_dev_procs.UnmapMemory(device_, mapped_memory_range.memory);
        }
    }


    VkCommandBuffer command_buffer = this_frame_resources->command_buffer;

    vk_dev_procs.ResetCommandBuffer(command_buffer, 0);


    GridPipelineFragmentShaderPushConstants grid_pipeline_frag_shader_push_constants {
        .world_to_screen_transform_inverse = *world_to_screen_transform_inverse, // OPTIMIZE this is a 64-byte copy, is that a lot?
        .viewport_offset_in_window = vec2(window_subregion.offset.x, window_subregion.offset.y),
        .viewport_size_in_window = vec2(window_subregion.extent.width, window_subregion.extent.height),
    };


    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    result = vk_dev_procs.BeginCommandBuffer(command_buffer, &begin_info);
    assertVk(result);
    {
        ZoneScopedN("cmd buf record");

        // TODO maybe we shouldn't hardcode this, if we're doing the whole "attached renderer" thing?
        // Maybe have a function pointer in the renderer or something to the appropriate Render function. Idk,
        // this is getting kinda weird. Maybe we should just ditch the whole generic crap.
        bool success = recordCommandBuffer(
            this_frame_resources,
            voxel_count,
            outlined_voxel_index_count,
            p_render_resources->render_pass,
            p_surface_resources->swapchain_extent,
            window_subregion,
            &grid_pipeline_frag_shader_push_constants,
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


    constexpr u32 wait_semaphore_count = 2;
    const VkSemaphore wait_semaphores[wait_semaphore_count] {
        swapchain_image_acquired_semaphore,
        p_surface_resources->swapchain_image_in_use_semaphores[acquired_swapchain_image_idx],
    };
    const VkPipelineStageFlags wait_dst_stage_mask[wait_semaphore_count] {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
    };

    constexpr u32 signal_semaphore_count = 2;
    const VkSemaphore signal_semaphores[signal_semaphore_count] {
        this_frame_resources->render_finished_semaphore,
        p_surface_resources->swapchain_image_in_use_semaphores[acquired_swapchain_image_idx],
    };

    const VkSubmitInfo submit_info {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = wait_semaphore_count,
        .pWaitSemaphores = wait_semaphores,
        .pWaitDstStageMask = wait_dst_stage_mask,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
        .signalSemaphoreCount = signal_semaphore_count,
        .pSignalSemaphores = signal_semaphores,
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
    {
        ZoneScopedN("queuePresent");
        result = vk_dev_procs.QueuePresentKHR(queue_, &present_info);
    }

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


extern void setGridEnabled(bool enable) {
    grid_enabled_ = enable;
}


extern bool setShaderSourceFileModificationTracking(bool enable) {

    if (enable == shader_source_file_watch_enabled_) return true;

    if (enable) {

        shader_source_file_watchlist_ = filewatch::createWatchlist();
        if (shader_source_file_watchlist_ == NULL) return false;

        for (u32fast pipeline_idx = 0; pipeline_idx < PIPELINE_INDEX_COUNT; pipeline_idx++) {

            const PipelineHotReloadInfo* p_source_info = &PIPELINE_HOT_RELOAD_INFOS[pipeline_idx];

            filewatch::FileID watch_id;
            
            watch_id = filewatch::addFileToModificationWatchlist(
                shader_source_file_watchlist_, p_source_info->vertex_shader_src_filepath
            );
            shader_source_file_watch_ids_[pipeline_idx].vertex_shader_id = watch_id;

            watch_id = filewatch::addFileToModificationWatchlist(
                shader_source_file_watchlist_, p_source_info->fragment_shader_src_filepath
            );
            shader_source_file_watch_ids_[pipeline_idx].fragment_shader_id = watch_id;
        }

        shader_source_file_watch_enabled_ = true;
    }
    else {
        filewatch::destroyWatchlist(shader_source_file_watchlist_);
        shader_source_file_watch_enabled_ = false;
    }

    return true;
};


// TODO Failure to compile a shader should not be fatal; this should at least return a bool indicating success
// or failure, without replacing the existing pipeline if compilation fails.
// TODO FIXME: this "passing around a renderer" shit isn't really working out, it's making things kinda
// weird. Why should the user pass in a renderer here? We're watching all the shader sources, not just the
// ones used by this renderer.
// The reason we're doing this is that pipelines are created for a particular render pass, subpass, and
// descriptor set layout. But then why are we storing these pipelines in their own global variables instead of
// in the renderer? They won't be compatible with other renderers.
// We should probably be storing the pipelines in the renderer. Although the descriptor set layout a pipeline
// is created with should probably be the same for all versions of the pipeline created in different renderers,
// so maybe that should be global? ugh. Maybe you need to sit down and try to draw out a dependency graph
// that includes RenderResources, pipelines, render passes, and descriptor set layouts.
extern ShaderReloadResult reloadModifiedShaderSourceFiles(RenderResources renderer) {

    ZoneScoped;

    assert(initialized_);
    assert(shader_source_file_watch_enabled_ && "Shader source file tracking is not enabled!");

    const RenderResourcesImpl* p_render_resources = (const RenderResourcesImpl*)renderer.impl;
    assert(p_render_resources != NULL);

    u32 event_count = 0;
    const filewatch::FileID* p_events = NULL;
    filewatch::poll(shader_source_file_watchlist_, &event_count, &p_events);

    if (event_count == 0) return ShaderReloadResult::no_shaders_need_reloading;


    timespec start_time {};
    {
        int success = timespec_get(&start_time, TIME_UTC);
        LOG_IF_F(ERROR, !success, "Failed to get shader rebuild start time.");
    }


    VkShaderStageFlags modified_shaders[PIPELINE_INDEX_COUNT] {};

    GraphicsPipelineShaderModules new_shader_modules[PIPELINE_INDEX_COUNT];
    memcpy(new_shader_modules, shader_modules_, PIPELINE_INDEX_COUNT * sizeof(GraphicsPipelineShaderModules));

    for (u32fast event_idx = 0; event_idx < event_count; event_idx++) {
        for (u32fast pipeline_idx = 0; pipeline_idx < PIPELINE_INDEX_COUNT; pipeline_idx++) {

            filewatch::FileID event_watch_id = p_events[event_idx];
            ShaderSourceFileWatchIds* this_pipeline_watch_ids = &shader_source_file_watch_ids_[pipeline_idx];

            const char* shader_src_filepath = NULL;
            shaderc_shader_kind shader_type;
            VkShaderModule* p_shader_module = NULL;

            if (this_pipeline_watch_ids->vertex_shader_id == event_watch_id) {
                shader_src_filepath =  PIPELINE_HOT_RELOAD_INFOS[pipeline_idx].vertex_shader_src_filepath;
                shader_type = shaderc_glsl_vertex_shader;
                p_shader_module = &new_shader_modules[pipeline_idx].vertex_shader_module;
                modified_shaders[pipeline_idx] |= VK_SHADER_STAGE_VERTEX_BIT;
            }
            else if (this_pipeline_watch_ids->fragment_shader_id == event_watch_id) {
                shader_src_filepath =  PIPELINE_HOT_RELOAD_INFOS[pipeline_idx].fragment_shader_src_filepath;
                shader_type = shaderc_glsl_fragment_shader;
                p_shader_module = &new_shader_modules[pipeline_idx].fragment_shader_module;
                modified_shaders[pipeline_idx] |= VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            else continue;

            LOG_F(
                 INFO, "Shader `%s` (pipeline idx %" PRIuFAST32 ") changed. Will reload.",
                 shader_src_filepath, pipeline_idx
            );

            bool success = createShaderModuleFromShaderSourceFile(
                device_, shader_src_filepath, shader_type, p_shader_module
            );
            if (!success) return ShaderReloadResult::error;
        }
    }


    PipelineAndLayout new_pipelines[PIPELINE_INDEX_COUNT];
    memcpy(new_pipelines, pipelines_, PIPELINE_INDEX_COUNT * sizeof(PipelineAndLayout));

    for (u32fast pipeline_idx = 0; pipeline_idx < PIPELINE_INDEX_COUNT; pipeline_idx++) {
        if (modified_shaders[pipeline_idx] != 0) {
            bool success = PIPELINE_HOT_RELOAD_INFOS[pipeline_idx].pfn_createPipeline(
                device_,
                new_shader_modules[pipeline_idx].vertex_shader_module,
                new_shader_modules[pipeline_idx].fragment_shader_module,
                p_render_resources->render_pass,
                the_only_subpass_,
                descriptor_set_layout_,
                &new_pipelines[pipeline_idx].pipeline,
                &new_pipelines[pipeline_idx].layout
            );
            if (!success) return ShaderReloadResult::error;
        }
    }


    VkResult result = vk_dev_procs.QueueWaitIdle(queue_);
    assertVk(result);

    for (u32fast pipeline_idx = 0; pipeline_idx < PIPELINE_INDEX_COUNT; pipeline_idx++) {

        VkShaderStageFlags this_pipeline_modified_shaders = modified_shaders[pipeline_idx];

        if (this_pipeline_modified_shaders != 0) {
            vk_dev_procs.DestroyPipeline(device_, pipelines_[pipeline_idx].pipeline, NULL);
            // OPTIMIZE we probably don't need to rebuild the layout
            vk_dev_procs.DestroyPipelineLayout(device_, pipelines_[pipeline_idx].layout, NULL);

            if (this_pipeline_modified_shaders & VK_SHADER_STAGE_VERTEX_BIT) vk_dev_procs.DestroyShaderModule(
                device_, shader_modules_[pipeline_idx].vertex_shader_module, NULL
            );
            if (this_pipeline_modified_shaders & VK_SHADER_STAGE_FRAGMENT_BIT) vk_dev_procs.DestroyShaderModule(
                device_, shader_modules_[pipeline_idx].fragment_shader_module, NULL
            );
        }
    }

    memcpy(shader_modules_, new_shader_modules, sizeof(shader_modules_));
    memcpy(pipelines_, new_pipelines, sizeof(pipelines_));


    timespec end_time;
    {
        int success = timespec_get(&end_time, TIME_UTC);
        LOG_IF_F(ERROR, !success, "Failed to get shader rebuild end time.");
    }

    f64 duration_milliseconds =
        (f64)(end_time.tv_sec - start_time.tv_sec) * 1'000. +
        (f64)(end_time.tv_nsec - start_time.tv_nsec) / 1'000'000.;
    LOG_F(INFO, "Shaders reloaded (%.0lf ms).", duration_milliseconds);


    return ShaderReloadResult::success;
}

//
// ===========================================================================================================
//

} // namespace

