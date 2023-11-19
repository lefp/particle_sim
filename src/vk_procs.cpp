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
    this->createFramebuffer = (PFN_vkCreateFramebuffer)getDeviceProcAddr(device, "vkCreateFramebuffer");
    this->createGraphicsPipelines = (PFN_vkCreateGraphicsPipelines)getDeviceProcAddr(device, "vkCreateGraphicsPipelines");
    this->createImageView = (PFN_vkCreateImageView)getDeviceProcAddr(device, "vkCreateImageView");
    this->createPipelineLayout = (PFN_vkCreatePipelineLayout)getDeviceProcAddr(device, "vkCreatePipelineLayout");
    this->createRenderPass = (PFN_vkCreateRenderPass)getDeviceProcAddr(device, "vkCreateRenderPass");
    this->createShaderModule = (PFN_vkCreateShaderModule)getDeviceProcAddr(device, "vkCreateShaderModule");
    this->createSwapchainKHR = (PFN_vkCreateSwapchainKHR)getDeviceProcAddr(device, "vkCreateSwapchainKHR");
    this->destroyShaderModule = (PFN_vkDestroyShaderModule)getDeviceProcAddr(device, "vkDestroyShaderModule");
    this->getDeviceQueue = (PFN_vkGetDeviceQueue)getDeviceProcAddr(device, "vkGetDeviceQueue");
    this->getSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)getDeviceProcAddr(device, "vkGetSwapchainImagesKHR");

    if (!allPointersNonNull(this)) ABORT_F("Some procedure pointers were not initialized.");
}
