#ifndef _VK_PROCS_HPP
#define _VK_PROCS_HPP

// #include <vulkan/vulkan.h>
// #include <loguru/loguru.hpp>

//
// ===========================================================================================================
//

/// These are where you define the lists of procedures you want to use.

#define FOR_EACH_VK_BASE_PROC(X) \
    X(CreateInstance) \
    X(GetInstanceProcAddr)

#define FOR_EACH_VK_INSTANCE_PROC(X) \
    X(CreateDevice) \
    X(EnumeratePhysicalDevices) \
    X(GetDeviceProcAddr) \
    X(GetPhysicalDeviceFeatures2) \
    X(GetPhysicalDeviceProperties) \
    X(GetPhysicalDeviceQueueFamilyProperties) \
    X(GetPhysicalDeviceSurfaceCapabilitiesKHR) \
    X(GetPhysicalDeviceSurfacePresentModesKHR)

#define FOR_EACH_VK_DEVICE_PROC(X) \
    X(AcquireNextImageKHR) \
    X(AllocateCommandBuffers) \
    X(AllocateDescriptorSets) \
    X(BeginCommandBuffer) \
    X(CmdBeginRendering) \
    X(CmdBindDescriptorSets) \
    X(CmdBindPipeline) \
    X(CmdBindIndexBuffer) \
    X(CmdBindVertexBuffers) \
    X(CmdCopyImage) \
    X(CmdDraw) \
    X(CmdDrawIndexed) \
    X(CmdEndRendering) \
    X(CmdPipelineBarrier) \
    X(CmdPushConstants) \
    X(CmdSetScissor) \
    X(CmdSetViewport) \
    X(CreateCommandPool) \
    X(CreateDescriptorPool) \
    X(CreateDescriptorSetLayout) \
    X(CreateFence) \
    X(CreateGraphicsPipelines) \
    X(CreateImageView) \
    X(CreatePipelineLayout) \
    X(CreateSemaphore) \
    X(CreateShaderModule) \
    X(CreateSwapchainKHR) \
    X(DestroyFence) \
    X(DestroyImageView) \
    X(DestroyPipeline) \
    X(DestroyPipelineLayout) \
    X(DestroySemaphore) \
    X(DestroyShaderModule) \
    X(DestroySwapchainKHR) \
    X(EndCommandBuffer) \
    X(FlushMappedMemoryRanges) \
    X(FreeCommandBuffers) \
    X(GetDeviceQueue) \
    X(GetFenceStatus) \
    X(GetSwapchainImagesKHR) \
    X(MapMemory) \
    X(QueuePresentKHR) \
    X(QueueSubmit) \
    X(QueueWaitIdle) \
    X(ResetCommandBuffer) \
    X(ResetCommandPool) \
    X(ResetFences) \
    X(UnmapMemory) \
    X(UpdateDescriptorSets) \
    X(WaitForFences)

//
// ===========================================================================================================
//

#define DECLARE_PROC_PTR(PROC_NAME) PFN_vk##PROC_NAME PROC_NAME;

struct VulkanBaseProcs {
    FOR_EACH_VK_BASE_PROC(DECLARE_PROC_PTR);
};

struct VulkanInstanceProcs {
    FOR_EACH_VK_INSTANCE_PROC(DECLARE_PROC_PTR);
};

struct VulkanDeviceProcs {
    FOR_EACH_VK_DEVICE_PROC(DECLARE_PROC_PTR);
};


#undef DECLARE_PROC_PTR

//
// ===========================================================================================================
//

#endif // include guard
