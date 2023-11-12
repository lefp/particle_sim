#include <vulkan/vulkan.h>

#include "vk_procs.hpp"
#include "alwaysAssert.hpp"

//
// ===========================================================================================================
//

void VulkanInstanceProcs::init(
    VkInstance instance,
    PFN_vkGetInstanceProcAddr getInstanceProcAddr
) {
    this->enumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)getInstanceProcAddr(instance, "vkEnumeratePhysicalDevices");
    alwaysAssert(this->enumeratePhysicalDevices != NULL);

    // TODO initialize the rest!
    alwaysAssert(false && "unimplemented");
}

void VulkanDeviceProcs::init(
    VkDevice device,
    PFN_vkGetDeviceProcAddr getDeviceProcAddr
) {
    // TODO
}
