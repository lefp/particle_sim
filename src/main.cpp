#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include "types.hpp"
#include "error_utils.hpp"
#include "vk_procs.hpp"
#include "defer.hpp"
#include "alloc_util.hpp"
#include "log_stub.hpp"
#include "error_utils.hpp"

//
// Global constants ==========================================================================================
//

#define VULKAN_API_VERSION VK_API_VERSION_1_3

const char* APP_NAME = "an game";

// Use to represent an invalid queue family; can't use -1 (because unsigned) or 0 (because it's valid).
const u32 INVALID_QUEUE_FAMILY_IDX = UINT32_MAX;

//
// Global variables ==========================================================================================
//

VkInstance instance_;
VkPhysicalDevice physical_device_;
VkPhysicalDeviceProperties physical_device_properties_;
u32 queue_family_;
VkDevice device_;

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

    logging::error(
        "Assertion failed! GLFW error code %i, file `%s`, line %u, description `%s`",
        err_code, file, line, err_description
    );
    abort();
};
#define assertGlfw(condition) _assertGlfw(condition, __FILE__, __LINE__)


void _abortIfGlfwError(const char* file, int line) {

    const char* err_description = NULL;
    int err_id = glfwGetError(&err_description);
    if (err_id == GLFW_NO_ERROR) return;

    if (err_description == NULL) err_description = "(NO DESCRIPTION PROVIDED)";
    logging::error(
        "Aborting due to GLFW error code %i, file `%s`, line %u, description `%s`",
        err_id, err_description, file, line
    );
    abort();
};
#define abortIfGlfwError() _abortIfGlfwError(__FILE__, __LINE__)


void _assertVk(VkResult result, const char* file, int line) {

    if (result == VK_SUCCESS) return;

    logging::error(
        "VkResult is %i, file `%s`, line %u",
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
            // TODO you shouldn't use %lu here because you don't know how large dev_idx is
            logging::info("Physical device `%lu` has no satisfactory queue family.", dev_idx);
            continue;
        }


        current_best_device = device;
        current_best_device_priority = device_priority;
        current_best_device_queue_family = fam;
    }

    *device_out = current_best_device;
    *queue_family_out = current_best_device_queue_family;
}


void initGraphics(void) {
    if (!glfwVulkanSupported()) logging::error("Failed to find Vulkan; do you need to install drivers?");
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

        if (physical_device_count == 0) abortWithMessage("Found no Vulkan devices.");
        VkPhysicalDevice* physical_devices = mallocArray<VkPhysicalDevice>(physical_device_count);
        defer(free(physical_devices));

        result = vk_inst_procs.enumeratePhysicalDevices(instance_, &physical_device_count, physical_devices);
        assertVk(result);

        for (u32 i = 0; i < physical_device_count; i++) {
            VkPhysicalDeviceProperties props;
            vk_inst_procs.getPhysicalDeviceProperties(physical_devices[i], &props);
            logging::info("Found physical device %u: `%s`", i, props.deviceName);
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
        logging::info("Selected physical device `%s`.", physical_device_properties_.deviceName);
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
    }
}


int main(void) {

    int success = glfwInit();
    assertGlfw(success);

    // TODO rename `initGraphics` to something like `initVulkanUptoQueueCreation` or something. Reason: we
    // may may want to have window and swapchain creation in a separate procedure, in case we want to
    // dynamically create multiple windows. Maybe pipeline creation should be separate too.
    initGraphics();

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // TODO: enable once swapchain resizing is implemented
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // don't initialize OpenGL, because we're using Vulkan
    GLFWwindow* window = glfwCreateWindow(800, 600, "an game", NULL, NULL);
    assertGlfw(window != NULL);


    glfwPollEvents();
    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();
    };


    glfwTerminate();
    exit(0);
}
