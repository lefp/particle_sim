#ifndef _DESCRIPTOR_MANAGEMENT_HPP
#define _DESCRIPTOR_MANAGEMENT_HPP

// #include <cassert>
// #include <vulkan/vulkan.h>
// #include <loguru/loguru.hpp>
// #include "types.hpp"
// #include "vulkan_context.hpp"
// #include "error_util.hpp"
// #include "math_util.hpp"
// #include "alloc_util.hpp"
// #include "defer.hpp"

namespace descriptor_management {

static_assert(VK_DESCRIPTOR_TYPE_SAMPLER == 0);
static_assert(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER == 1);
static_assert(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE == 2);
static_assert(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE == 3);
static_assert(VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER == 4);
static_assert(VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER == 5);
static_assert(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER == 6);
static_assert(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER == 7);
static_assert(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC == 8);
static_assert(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC == 9);
static_assert(VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT == 10);
// For simplicity, we cap it at 10, because the enum values for the other descriptor types are huge and cannot
// be used as array indices.
constexpr u32 MAX_SUPPORTED_DESCRIPTOR_TYPE = 10;

//
// ===========================================================================================================
//

struct DescriptorSetLayout {
    u32 binding_count;
    VkDescriptorSetLayoutBinding* p_bindings;
};

void createDescriptorPoolAndSets(

    const VulkanContext* vk_ctx,

    const u32fast layout_count,
    const DescriptorSetLayout *const p_layouts,
    const u32 *const p_set_counts, // how many descriptor sets to allocate with each layout

    VkDescriptorPool *const p_descriptor_pool_out,
    VkDescriptorSetLayout *const p_descriptor_set_layouts_out, // length = layout_count
    VkDescriptorSet *const p_descriptor_sets_out // length = sum(p_alloc_counts)
);

//
// ===========================================================================================================
//

} // namespace

#endif // include guard
