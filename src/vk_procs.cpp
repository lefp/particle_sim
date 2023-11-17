#include <vulkan/vulkan.h>
#include <loguru.hpp>

#include "types.hpp"
#include "vk_procs.hpp"
#include "error_utils.hpp"

//
// ===========================================================================================================
//

VulkanInstanceProcs vk_inst_procs {};
VulkanDeviceProcs vk_dev_procs {};

//
// ===========================================================================================================
//

template <typename StructOfPointers>
static bool allPointersNonNull(const StructOfPointers* s) {

    constexpr u32fast ptr_count = sizeof(StructOfPointers) / sizeof(void*);
    const void** ptrs = (const void**)s;

    for (u32fast i = 0; i < ptr_count; i++) if (ptrs[i] == NULL) return false;
    return true;
}

void VulkanInstanceProcs::init(
    VkInstance instance,
    PFN_vkGetInstanceProcAddr getInstanceProcAddr
) {
    this->createDevice = (PFN_vkCreateDevice)getInstanceProcAddr(instance, "vkCreateDevice");
    this->enumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)getInstanceProcAddr(instance, "vkEnumeratePhysicalDevices");
    this->getDeviceProcAddr = (PFN_vkGetDeviceProcAddr)getInstanceProcAddr(instance, "vkGetDeviceProcAddr");
    this->getPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)getInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties");
    this->getPhysicalDeviceQueueFamilyProperties = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)getInstanceProcAddr(instance, "vkGetPhysicalDeviceQueueFamilyProperties");

    if (!allPointersNonNull(this)) ABORT_F("Some procedure pointers were not initialized.");
}

void VulkanDeviceProcs::init(
    VkDevice device,
    PFN_vkGetDeviceProcAddr getDeviceProcAddr
) {
    this->getDeviceQueue = (PFN_vkGetDeviceQueue)getDeviceProcAddr(device, "vkGetDeviceQueue");

    if (!allPointersNonNull(this)) ABORT_F("Some procedure pointers were not initialized.");
}
