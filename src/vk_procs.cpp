#include <vulkan/vulkan.h>

#include "types.hpp"
#include "vk_procs.hpp"
#include "log_stub.hpp"
#include "error_utils.hpp"

//
// ===========================================================================================================
//

VulkanInstanceProcs vk_inst_procs {};
VulkanDeviceProcs vk_dev_procs {};

//
// ===========================================================================================================
//

void VulkanInstanceProcs::init(
    VkInstance instance,
    PFN_vkGetInstanceProcAddr getInstanceProcAddr
) {
    this->createDevice = (PFN_vkCreateDevice)getInstanceProcAddr(instance, "vkCreateDevice");
    this->enumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)getInstanceProcAddr(instance, "vkEnumeratePhysicalDevices");
    this->getDeviceProcAddr = (PFN_vkGetDeviceProcAddr)getInstanceProcAddr(instance, "vkGetDeviceProcAddr");
    this->getPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)getInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties");
    this->getPhysicalDeviceQueueFamilyProperties = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)getInstanceProcAddr(instance, "vkGetPhysicalDeviceQueueFamilyProperties");

    // verify that we've initialized all procedure pointers
    constexpr u32fast proc_pointer_count = sizeof(VulkanInstanceProcs) / sizeof(void*);
    const void** proc_pointers = (const void**)this;
    bool all_ptrs_initialized = true;
    for (u32fast i = 0; i < proc_pointer_count; i++) {
        if (proc_pointers[i] == NULL) {
            // TODO you're using %lu, but you don't know the sizeof u32fast. This is a bug.
            logging::error("Procedure pointer at index %lu was not initialized.", i);
            all_ptrs_initialized = false;
        }
    }
    if (!all_ptrs_initialized) abortWithMessage("Some procedure pointers were not initialized.");
}

void VulkanDeviceProcs::init(
    VkDevice device,
    PFN_vkGetDeviceProcAddr getDeviceProcAddr
) {
    // TODO
}
