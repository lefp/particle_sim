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
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR getPhysicalDeviceSurfaceCapabilitiesKHR;

    void init(VkInstance, PFN_vkGetInstanceProcAddr);
};

struct VulkanDeviceProcs {
    PFN_vkCreateFramebuffer createFramebuffer;
    PFN_vkCreateGraphicsPipelines createGraphicsPipelines;
    PFN_vkCreateImageView createImageView;
    PFN_vkCreatePipelineLayout createPipelineLayout;
    PFN_vkCreateRenderPass createRenderPass;
    PFN_vkCreateShaderModule createShaderModule;
    PFN_vkCreateSwapchainKHR createSwapchainKHR;
    PFN_vkDestroyShaderModule destroyShaderModule;
    PFN_vkGetDeviceQueue getDeviceQueue;
    PFN_vkGetSwapchainImagesKHR getSwapchainImagesKHR;

    void init(VkDevice, PFN_vkGetDeviceProcAddr);
};

//
// ===========================================================================================================
//

extern VulkanInstanceProcs vk_inst_procs;
extern VulkanDeviceProcs vk_dev_procs;
