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
    this->getPhysicalDeviceSurfaceCapabilitiesKHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)getInstanceProcAddr(instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    if (!allPointersNonNull(this)) ABORT_F("Some procedure pointers were not initialized.");
}

void VulkanDeviceProcs::init(
    VkDevice device,
    PFN_vkGetDeviceProcAddr getDeviceProcAddr
) {
    this->acquireNextImageKHR = (PFN_vkAcquireNextImageKHR)getDeviceProcAddr(device, "vkAcquireNextImageKHR");
    this->allocateCommandBuffers = (PFN_vkAllocateCommandBuffers)getDeviceProcAddr(device, "vkAllocateCommandBuffers");
    this->beginCommandBuffer = (PFN_vkBeginCommandBuffer)getDeviceProcAddr(device, "vkBeginCommandBuffer");
    this->cmdBeginRenderPass = (PFN_vkCmdBeginRenderPass)getDeviceProcAddr(device, "vkCmdBeginRenderPass");
    this->cmdBindPipeline = (PFN_vkCmdBindPipeline)getDeviceProcAddr(device, "vkCmdBindPipeline");
    this->cmdDraw = (PFN_vkCmdDraw)getDeviceProcAddr(device, "vkCmdDraw");
    this->cmdEndRenderPass = (PFN_vkCmdEndRenderPass)getDeviceProcAddr(device, "vkCmdEndRenderPass");
    this->cmdSetScissor = (PFN_vkCmdSetScissor)getDeviceProcAddr(device, "vkCmdSetScissor");
    this->cmdSetViewport = (PFN_vkCmdSetViewport)getDeviceProcAddr(device, "vkCmdSetViewport");
    this->createCommandPool = (PFN_vkCreateCommandPool)getDeviceProcAddr(device, "vkCreateCommandPool");
    this->createFence = (PFN_vkCreateFence)getDeviceProcAddr(device, "vkCreateFence");
    this->createFramebuffer = (PFN_vkCreateFramebuffer)getDeviceProcAddr(device, "vkCreateFramebuffer");
    this->createGraphicsPipelines = (PFN_vkCreateGraphicsPipelines)getDeviceProcAddr(device, "vkCreateGraphicsPipelines");
    this->createImageView = (PFN_vkCreateImageView)getDeviceProcAddr(device, "vkCreateImageView");
    this->createPipelineLayout = (PFN_vkCreatePipelineLayout)getDeviceProcAddr(device, "vkCreatePipelineLayout");
    this->createRenderPass = (PFN_vkCreateRenderPass)getDeviceProcAddr(device, "vkCreateRenderPass");
    this->createSemaphore = (PFN_vkCreateSemaphore)getDeviceProcAddr(device, "vkCreateSemaphore");
    this->createShaderModule = (PFN_vkCreateShaderModule)getDeviceProcAddr(device, "vkCreateShaderModule");
    this->createSwapchainKHR = (PFN_vkCreateSwapchainKHR)getDeviceProcAddr(device, "vkCreateSwapchainKHR");
    this->destroyFramebuffer = (PFN_vkDestroyFramebuffer)getDeviceProcAddr(device, "vkDestroyFramebuffer");
    this->destroyImageView = (PFN_vkDestroyImageView)getDeviceProcAddr(device, "vkDestroyImageView");
    this->destroyShaderModule = (PFN_vkDestroyShaderModule)getDeviceProcAddr(device, "vkDestroyShaderModule");
    this->endCommandBuffer = (PFN_vkEndCommandBuffer)getDeviceProcAddr(device, "vkEndCommandBuffer");
    this->getDeviceQueue = (PFN_vkGetDeviceQueue)getDeviceProcAddr(device, "vkGetDeviceQueue");
    this->getSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)getDeviceProcAddr(device, "vkGetSwapchainImagesKHR");
    this->queuePresentKHR = (PFN_vkQueuePresentKHR)getDeviceProcAddr(device, "vkQueuePresentKHR");
    this->queueSubmit = (PFN_vkQueueSubmit)getDeviceProcAddr(device, "vkQueueSubmit");
    this->resetCommandBuffer = (PFN_vkResetCommandBuffer)getDeviceProcAddr(device, "vkResetCommandBuffer");
    this->resetFences = (PFN_vkResetFences)getDeviceProcAddr(device, "vkResetFences");
    this->waitForFences = (PFN_vkWaitForFences)getDeviceProcAddr(device, "vkWaitForFences");

    if (!allPointersNonNull(this)) ABORT_F("Some procedure pointers were not initialized.");
}
