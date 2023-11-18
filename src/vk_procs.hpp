// #include <vulkan/vulkan.h>

//
// ===========================================================================================================
//

struct VulkanInstanceProcs {
    PFN_vkCreateDevice createDevice;
    PFN_vkEnumeratePhysicalDevices enumeratePhysicalDevices;
    PFN_vkGetDeviceProcAddr getDeviceProcAddr;
    PFN_vkGetPhysicalDeviceProperties getPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties getPhysicalDeviceQueueFamilyProperties;

    void init(VkInstance, PFN_vkGetInstanceProcAddr);
};

struct VulkanDeviceProcs {
    PFN_vkCreateGraphicsPipelines createGraphicsPipelines;
    PFN_vkCreatePipelineLayout createPipelineLayout;
    PFN_vkCreateRenderPass createRenderPass;
    PFN_vkCreateShaderModule createShaderModule;
    PFN_vkDestroyShaderModule destroyShaderModule;
    PFN_vkGetDeviceQueue getDeviceQueue;

    void init(VkDevice, PFN_vkGetDeviceProcAddr);
};

//
// ===========================================================================================================
//

extern VulkanInstanceProcs vk_inst_procs;
extern VulkanDeviceProcs vk_dev_procs;
