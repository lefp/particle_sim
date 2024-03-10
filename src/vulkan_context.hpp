#ifndef _VULKAN_CONTEXT_HPP
#define _VULKAN_CONTEXT_HPP

// #include <VulkanMemoryAllocator/vk_mem_alloc.h>

// #include "types.hpp"
// #include "vk_procs.hpp"

//
// ===========================================================================================================
//

struct VulkanContext {

    VulkanBaseProcs procs_base;
    VulkanInstanceProcs procs_inst;
    VulkanDeviceProcs procs_dev;

    VmaAllocator vma_allocator;

    VkDevice device;
    u32 queue_family_index;
    VkQueue queue;

    VkPhysicalDeviceProperties physical_device_properties;
};

//
// ===========================================================================================================
//

#endif // include guard
