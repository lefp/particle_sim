#include <cassert>

#include <vulkan/vulkan.h>
#include <loguru/loguru.hpp>

#include "types.hpp"
#include "vulkan_context.hpp"
#include "error_util.hpp"
#include "math_util.hpp"
#include "alloc_util.hpp"
#include "defer.hpp"
#include "descriptor_management.hpp"

namespace descriptor_management {

//
// ===========================================================================================================
//

static void _assertVk(VkResult result, const char* file, int line) {

    if (result == VK_SUCCESS) return;

    ABORT_F("VkResult is %i, file `%s`, line %i", result, file, line);
}
#define assertVk(result) _assertVk(result, __FILE__, __LINE__)


extern void createDescriptorPoolAndSets(

    const VulkanContext* vk_ctx,

    const u32fast layout_count,
    const DescriptorSetLayout *const p_layouts,
    const u32 *const p_set_counts, // how many descriptor sets to allocate with each layout

    VkDescriptorPool *const p_descriptor_pool_out,
    VkDescriptorSetLayout *const p_descriptor_set_layouts_out, // [layout_count]
    VkDescriptorSet *const p_descriptor_sets_out // sum(p_alloc_counts)
) {
    assert(layout_count > 0);
    assert(p_layouts != NULL);

    // indexed by VkDescriptorType
    u32 descriptor_type_counts[MAX_SUPPORTED_DESCRIPTOR_TYPE + 1] {};
    {
        for (u32fast layout_idx = 0; layout_idx < layout_count; layout_idx++)
        {
            const DescriptorSetLayout* layout = &p_layouts[layout_idx];
            const u32 alloc_count = p_set_counts[layout_idx];
            assert(alloc_count > 0);

            assert(layout->binding_count > 0);
            for (u32fast binding_idx = 0; binding_idx < layout->binding_count; binding_idx++)
            {
                VkDescriptorType descriptor_type = layout->p_bindings[binding_idx].descriptorType;
                assert(descriptor_type <= MAX_SUPPORTED_DESCRIPTOR_TYPE);
                descriptor_type_counts[descriptor_type] += alloc_count;
            }
        }
    }


    u32fast unique_descriptor_type_count = 0;
    VkDescriptorPoolSize pool_sizes[MAX_SUPPORTED_DESCRIPTOR_TYPE + 1] {};
    {
        for (u32fast desc_type = 0; desc_type <= MAX_SUPPORTED_DESCRIPTOR_TYPE; desc_type++)
        {
            const u32 desc_type_count = descriptor_type_counts[desc_type];
            if (desc_type_count != 0)
            {
                pool_sizes[unique_descriptor_type_count] = VkDescriptorPoolSize {
                    .type = (VkDescriptorType)desc_type,
                    .descriptorCount = desc_type_count,
                };
                unique_descriptor_type_count++;
            }
        }
    }
    assert(unique_descriptor_type_count > 0);


    u32fast descriptor_set_count = 0;
    for (u32fast i = 0; i < layout_count; i++) descriptor_set_count += p_set_counts[i];
    assert(descriptor_set_count > 0);


    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    {
        VkDescriptorPoolCreateInfo pool_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = (u32)descriptor_set_count,
            .poolSizeCount = (u32)unique_descriptor_type_count,
            .pPoolSizes = pool_sizes,
        };
        VkResult result = vk_ctx->procs_dev.CreateDescriptorPool(
            vk_ctx->device, &pool_info, NULL, &descriptor_pool
        );
        assertVk(result);
    }
    *p_descriptor_pool_out = descriptor_pool;

    for (u32fast layout_idx = 0; layout_idx < layout_count; layout_idx++)
    {
        const DescriptorSetLayout* p_layout = &p_layouts[layout_idx];
        VkDescriptorSetLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = p_layout->binding_count,
            .pBindings = p_layout->p_bindings,
        };
        VkResult result = vk_ctx->procs_dev.CreateDescriptorSetLayout(
            vk_ctx->device,
            &layout_info,
            NULL,
            &p_descriptor_set_layouts_out[layout_idx]
        );
        assertVk(result);
    }

    {
        VkDescriptorSetLayout* descriptor_set_layouts = callocArray(descriptor_set_count, VkDescriptorSetLayout);
        defer(free(descriptor_set_layouts));

        u32fast descriptor_set_idx = 0;
        for (u32fast layout_idx = 0; layout_idx < layout_count; layout_idx++)
        {
            const u32fast set_count = p_set_counts[layout_idx];

            for (u32fast i = 0; i < set_count; i++)
            {
                descriptor_set_layouts[descriptor_set_idx] = p_descriptor_set_layouts_out[layout_idx];
                descriptor_set_idx++;
            }
        }

        VkDescriptorSetAllocateInfo alloc_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = (u32)descriptor_set_count,
            .pSetLayouts = descriptor_set_layouts,
        };
        VkResult result = vk_ctx->procs_dev.AllocateDescriptorSets(
            vk_ctx->device, &alloc_info, p_descriptor_sets_out
        );
        assertVk(result);

        for (u32fast i = 0; i < descriptor_set_count; i++)
        {
            LOG_F(
                INFO, "Allocated descriptor set %p, using layout %p.",
                p_descriptor_sets_out[i], descriptor_set_layouts[i]
            );
            fflush(stdout);
        }
    }
}

//
// ===========================================================================================================
//

}
