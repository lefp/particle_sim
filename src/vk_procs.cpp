#include <vulkan/vulkan.h>
#include <loguru/loguru.hpp>

#include "types.hpp"
#include "vk_procs.hpp"

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
    #define INITIALIZE_PROC_PTR(PROC_NAME) \
        { \
            const char* proc_name = "vk" #PROC_NAME; \
            this->PROC_NAME = (PFN_vk##PROC_NAME)getInstanceProcAddr(instance, proc_name); \
            if (this->PROC_NAME == NULL) ABORT_F("Failed to load instance procedure `%s`.", proc_name); \
        }

    FOR_EACH_INSTANCE_PROC(INITIALIZE_PROC_PTR);

    #undef INITIALIZE_PROC_PTR
}

void VulkanDeviceProcs::init(
    VkDevice device,
    PFN_vkGetDeviceProcAddr getDeviceProcAddr
) {
    #define INITIALIZE_PROC_PTR(PROC_NAME) \
        { \
            const char* proc_name = "vk" #PROC_NAME; \
            this->PROC_NAME = (PFN_vk##PROC_NAME)getDeviceProcAddr(device, proc_name); \
            if (this->PROC_NAME == NULL) ABORT_F("Failed to load device procedure `%s`.", proc_name); \
        }

    FOR_EACH_DEVICE_PROC(INITIALIZE_PROC_PTR);

    #undef INITIALIZE_PROC_PTR
}
