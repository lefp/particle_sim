#include <vulkan/vulkan.h>
#include <loguru/loguru.hpp>

#include "vk_procs.hpp"

//
// ===========================================================================================================
//

VulkanBaseProcs vk_base_procs {};
VulkanInstanceProcs vk_inst_procs {};
VulkanDeviceProcs vk_dev_procs {};

//
// ===========================================================================================================
//

void VulkanBaseProcs::init(
    PFN_vkGetInstanceProcAddr getInstanceProcAddr
) {
    #define INITIALIZE_PROC_PTR(PROC_NAME) \
        { \
            const char* proc_name = "vk" #PROC_NAME; \
            this->PROC_NAME = (PFN_vk##PROC_NAME)getInstanceProcAddr(NULL, proc_name); \
            if (this->PROC_NAME == NULL) ABORT_F("Failed to load base procedure `%s`.", proc_name); \
        }

    FOR_EACH_VK_BASE_PROC(INITIALIZE_PROC_PTR);

    #undef INITIALIZE_PROC_PTR
}

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

    FOR_EACH_VK_INSTANCE_PROC(INITIALIZE_PROC_PTR);

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

    FOR_EACH_VK_DEVICE_PROC(INITIALIZE_PROC_PTR);

    #undef INITIALIZE_PROC_PTR
}
