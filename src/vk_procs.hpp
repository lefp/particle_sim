#ifndef _VK_PROCS_HPP
#define _VK_PROCS_HPP

#include <vulkan/vulkan.h>
#include <loguru/loguru.hpp>

//
// ===========================================================================================================
//

/// These are where you define the lists of procedures you want to use.

#define FOR_EACH_BASE_PROC(X) \
    X(CreateInstance) \
    X(GetInstanceProcAddr)

#define FOR_EACH_INSTANCE_PROC(X) \
    X(CreateDevice) \
    X(EnumeratePhysicalDevices) \
    X(GetDeviceProcAddr) \
    X(GetPhysicalDeviceProperties) \
    X(GetPhysicalDeviceQueueFamilyProperties) \
    X(GetPhysicalDeviceSurfaceCapabilitiesKHR)

#define FOR_EACH_DEVICE_PROC(X) \
    X(AcquireNextImageKHR) \
    X(AllocateCommandBuffers) \
    X(BeginCommandBuffer) \
    X(CmdBeginRenderPass) \
    X(CmdBindPipeline) \
    X(CmdDraw) \
    X(CmdEndRenderPass) \
    X(CmdPipelineBarrier) \
    X(CmdPushConstants) \
    X(CmdSetScissor) \
    X(CmdSetViewport) \
    X(CreateCommandPool) \
    X(CreateFence) \
    X(CreateFramebuffer) \
    X(CreateGraphicsPipelines) \
    X(CreateImageView) \
    X(CreatePipelineLayout) \
    X(CreateRenderPass) \
    X(CreateSemaphore) \
    X(CreateShaderModule) \
    X(CreateSwapchainKHR) \
    X(DestroyFence) \
    X(DestroyFramebuffer) \
    X(DestroyImageView) \
    X(DestroySemaphore) \
    X(DestroyShaderModule) \
    X(DestroySwapchainKHR) \
    X(EndCommandBuffer) \
    X(FreeCommandBuffers) \
    X(GetDeviceQueue) \
    X(GetFenceStatus) \
    X(GetSwapchainImagesKHR) \
    X(QueuePresentKHR) \
    X(QueueSubmit) \
    X(QueueWaitIdle) \
    X(ResetCommandBuffer) \
    X(ResetCommandPool) \
    X(ResetFences) \
    X(WaitForFences)

//
// ===========================================================================================================
//

#define DECLARE_PROC_PTR(PROC_NAME) PFN_vk##PROC_NAME PROC_NAME;

struct VulkanBaseProcs {
    FOR_EACH_BASE_PROC(DECLARE_PROC_PTR);

    void init(PFN_vkGetInstanceProcAddr);
};

struct VulkanInstanceProcs {
    FOR_EACH_INSTANCE_PROC(DECLARE_PROC_PTR);

    void init(VkInstance, PFN_vkGetInstanceProcAddr);
};

struct VulkanDeviceProcs {
    FOR_EACH_DEVICE_PROC(DECLARE_PROC_PTR);

    void init(VkDevice, PFN_vkGetDeviceProcAddr);
};


#undef DECLARE_PROC_PTR

//
// ===========================================================================================================
//

extern VulkanBaseProcs vk_base_procs;
extern VulkanInstanceProcs vk_inst_procs;
extern VulkanDeviceProcs vk_dev_procs;

//
// ===========================================================================================================
//

#endif // include guard
