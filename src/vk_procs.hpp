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
    PFN_vkAcquireNextImageKHR acquireNextImageKHR;
    PFN_vkAllocateCommandBuffers allocateCommandBuffers;
    PFN_vkBeginCommandBuffer beginCommandBuffer;
    PFN_vkCmdBeginRenderPass cmdBeginRenderPass;
    PFN_vkCmdBindPipeline cmdBindPipeline;
    PFN_vkCmdDraw cmdDraw;
    PFN_vkCmdEndRenderPass cmdEndRenderPass;
    PFN_vkCmdSetScissor cmdSetScissor;
    PFN_vkCmdSetViewport cmdSetViewport;
    PFN_vkCreateCommandPool createCommandPool;
    PFN_vkCreateFence createFence;
    PFN_vkCreateFramebuffer createFramebuffer;
    PFN_vkCreateGraphicsPipelines createGraphicsPipelines;
    PFN_vkCreateImageView createImageView;
    PFN_vkCreatePipelineLayout createPipelineLayout;
    PFN_vkCreateRenderPass createRenderPass;
    PFN_vkCreateSemaphore createSemaphore;
    PFN_vkCreateShaderModule createShaderModule;
    PFN_vkCreateSwapchainKHR createSwapchainKHR;
    PFN_vkDestroyFramebuffer destroyFramebuffer;
    PFN_vkDestroyImageView destroyImageView;
    PFN_vkDestroyShaderModule destroyShaderModule;
    PFN_vkEndCommandBuffer endCommandBuffer;
    PFN_vkGetDeviceQueue getDeviceQueue;
    PFN_vkGetSwapchainImagesKHR getSwapchainImagesKHR;
    PFN_vkQueuePresentKHR queuePresentKHR;
    PFN_vkQueueSubmit queueSubmit;
    PFN_vkResetCommandBuffer resetCommandBuffer;
    PFN_vkResetFences resetFences;
    PFN_vkWaitForFences waitForFences;

    void init(VkDevice, PFN_vkGetDeviceProcAddr);
};

//
// ===========================================================================================================
//

extern VulkanInstanceProcs vk_inst_procs;
extern VulkanDeviceProcs vk_dev_procs;
