#include <cstring>
#include <cstdlib>
#include <cinttypes>
#include <cmath>
#include <x86intrin.h>

#define GLM_FORCE_EXPLICIT_CTOR
#include <glm/glm.hpp>
#include <loguru/loguru.hpp>
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <VulkanMemoryAllocator/vk_mem_alloc.h>
#include <tracy/tracy/Tracy.hpp>
#include <tracy/tracy/TracyVulkan.hpp>

#include "../src/types.hpp"
#include "../src/error_util.hpp"
#include "../src/math_util.hpp"
#include "../src/alloc_util.hpp"
#include "../src/file_util.hpp"
#include "../src/vk_procs.hpp"
#include "../src/vulkan_context.hpp"
#include "../src/defer.hpp"
#include "../src/thread_pool.hpp"
#include "../src/sort.hpp"
#include "../src/thread_pool.hpp"
#include "../src/descriptor_management.hpp"
#include "fluid_sim_types.hpp"

namespace fluid_sim {

using glm::vec3;
using glm::vec4;
using glm::uvec3;

//
// ===========================================================================================================
//

constexpr f32 PI = (f32)M_PI;

constexpr struct {
    u32 local_size_x = 0;
} COMPUTE_SHADER_SPECIALIZATION_CONSTANT_IDS;

//
// descriptor set layouts ====================================================================================
//

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))

enum DescriptorSetLayoutBinding_Reduction {
    LAYOUT_BINDING_REDUCTION__POSITIONS_IN = 0,
    LAYOUT_BINDING_REDUCTION__POSITIONS_OUT = 1,

    LAYOUT_BINDING_COUNT__REDUCTION
};
constexpr VkDescriptorType DESCRIPTOR_SET_LAYOUT__REDUCTION[] {
    [LAYOUT_BINDING_REDUCTION__POSITIONS_IN] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // std430
    [LAYOUT_BINDING_REDUCTION__POSITIONS_OUT] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // std430
};
static_assert(ARRAY_SIZE(DESCRIPTOR_SET_LAYOUT__REDUCTION) == LAYOUT_BINDING_COUNT__REDUCTION);

enum DescriptorSetLayoutBindings_General {
    LAYOUT_BINDING_GENERAL__UNIFORMS = 0,
    LAYOUT_BINDING_GENERAL__POSITIONS_SORTED = 1,
    LAYOUT_BINDING_GENERAL__VELOCITIES_SORTED = 2,
    LAYOUT_BINDING_GENERAL__POSITIONS_UNSORTED = 3,
    LAYOUT_BINDING_GENERAL__VELOCITIES_UNSORTED = 4,
    LAYOUT_BINDING_GENERAL__C_BEGIN = 5,
    LAYOUT_BINDING_GENERAL__C_LENGTH = 6,
    LAYOUT_BINDING_GENERAL__H_BEGIN = 7,
    LAYOUT_BINDING_GENERAL__H_LENGTH = 8,
    LAYOUT_BINDING_GENERAL__MORTON_CODES_OR_PERMUTATION = 9,

    LAYOUT_BINDING_COUNT__GENERAL
};
constexpr VkDescriptorType DESCRIPTOR_SET_LAYOUT__GENERAL[] {
    [LAYOUT_BINDING_GENERAL__UNIFORMS] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, // std140
    [LAYOUT_BINDING_GENERAL__POSITIONS_SORTED] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // std430, vec3[particle_count]
    [LAYOUT_BINDING_GENERAL__VELOCITIES_SORTED] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // std430, vec3[particle_count]
    [LAYOUT_BINDING_GENERAL__POSITIONS_UNSORTED] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // std430, vec3[particle_count]
    [LAYOUT_BINDING_GENERAL__VELOCITIES_UNSORTED] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // std430, vec3[particle_count]
    [LAYOUT_BINDING_GENERAL__C_BEGIN] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // std430, u32[particle_count]
    [LAYOUT_BINDING_GENERAL__C_LENGTH] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // std430, u32[particle_count
    [LAYOUT_BINDING_GENERAL__H_BEGIN] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // std430, u32[particle_count]
    [LAYOUT_BINDING_GENERAL__H_LENGTH] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // std430, u32[particle_count]
    [LAYOUT_BINDING_GENERAL__MORTON_CODES_OR_PERMUTATION] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // std430, u32[particle_count]
};
static_assert(ARRAY_SIZE(DESCRIPTOR_SET_LAYOUT__GENERAL) == LAYOUT_BINDING_COUNT__GENERAL);

//
// ===========================================================================================================
//

#define SWAP(a, b) \
{ \
    typeof(a) _tmp = a; \
    a = b; \
    b = _tmp; \
}


static void _assertVk(VkResult result, const char* file, int line) {

    if (result == VK_SUCCESS) return;

    ABORT_F("VkResult is %i, file `%s`, line %i", result, file, line);
}
#define assertVk(result) _assertVk(result, __FILE__, __LINE__)


static inline u32 divCeil(u32 numerator, u32 denominator) {
    return (numerator / denominator) + (numerator % denominator != 0);
}


/// Get the index of the cell that contains the particle.
static inline uvec3 cellIndex(vec3 particle, vec3 domain_min, f32 cell_size_reciprocal) {

    return uvec3((particle - domain_min) * cell_size_reciprocal);
}


// TODO FIXME OPTIMIZE:
//     1. This will not work on ARM.
//     2. Even on x86, this will only work with CPUs supporting BMI2.
//     3. On AMD Zen 1 and Zen 2, PDEP is slow as shit. https://fgiesen.wordpress.com/2022/09/09/morton-codes-addendum/
static inline u32 cellMortonCode(uvec3 cell_index) {

    // OPTIMIZE: if you do this using inline assembly instead of the intrinsics, you can use 1 less register
    // and avoid the `or`s, if the compiler isn't already smart enough to do that.

    u32 result = 0;
    result |= _pdep_u32(cell_index.x, 0b00'001001001001001001001001001001);
    result |= _pdep_u32(cell_index.y, 0b00'010010010010010010010010010010);
    result |= _pdep_u32(cell_index.z, 0b00'100100100100100100100100100100);
    return result;
}


static inline u32 mortonCodeHash(u32 cell_morton_code, u32 hash_modulus) {
    return cell_morton_code % hash_modulus;
}


struct ParticleUpdatePushConstants {
    alignas(4) f32 delta_t;
    alignas(4) u32 cell_count;
};

struct ReductionPushConstants {
    alignas(4) u32 array_size;
};

struct UniformBufferData {

    // stuff that may change every frame
    alignas(16) vec3 domain_min;

    alignas(4) struct UpdatedByHost {

        // stuff whose lifetime is the lifetime of the sim parameters
        alignas(4) f32 rest_particle_density;
        alignas(4) f32 particle_interaction_radius;
        alignas(4) f32 spring_rest_length;
        alignas(4) f32 spring_stiffness;
        alignas(4) f32 cell_size_reciprocal;

        // stuff whose lifetime is the lifetime of the sim
        alignas(4) u32 particle_count;
        alignas(4) u32 hash_modulus;
    } updated_by_host;
};


struct PipelineBarrierSrcInfo {
    VkPipelineStageFlags src_stage_mask;
    VkPipelineStageFlags src_access_mask;
};
[[nodiscard]] static PipelineBarrierSrcInfo recordMinReductionCommands(
    const SimData* s,
    const VulkanContext* vk_ctx,
    const VkCommandBuffer command_buffer
) {

    ZoneScoped;


    u32 array_length = (u32)s->particle_count;
    VkDescriptorSet descriptor_set = s->gpu_resources.descriptor_set_reduction__positions_to_reduction1;
    VkBuffer dst_buffer = s->gpu_resources.buffer_reduction_1.buffer;

    while (true)
    {
        vk_ctx->procs_dev.CmdBindPipeline(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            s->gpu_resources.pipeline_computeMin
        );
        vk_ctx->procs_dev.CmdBindDescriptorSets(
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            s->gpu_resources.pipeline_layout_computeMin,
            0, // firstSet
            1, // descriptorSetCount
            &descriptor_set,
            0, // dynamicOffsetCount
            NULL // pDynamicOffsets
        );

        const ReductionPushConstants push_constants { .array_size = array_length };
        vk_ctx->procs_dev.CmdPushConstants(
            command_buffer,
            s->gpu_resources.pipeline_layout_computeMin,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, // offset
            sizeof(ReductionPushConstants), // size
            &push_constants
        );
        // OPTIMIZE: Maybe should tune a special workgroup size for this pipeline, since the optimal
        //     parameters for this might not be the same as the optimal parameters for other GPU work.
        vk_ctx->procs_dev.CmdDispatch(command_buffer, s->gpu_resources.workgroup_count, 1, 1);

        array_length = divCeil(array_length, 2);

        // ---------------------------------------------------------------------------------------------------
        if (array_length == 1) break;
        // ---------------------------------------------------------------------------------------------------

        const VkBufferMemoryBarrier buffer_memory_barrier {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = vk_ctx->queue_family_index,
            .dstQueueFamilyIndex = vk_ctx->queue_family_index,
            .buffer = dst_buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE, // OPTIMIZE this can be smaller: array_size * sizeof(vec4)
        };
        vk_ctx->procs_dev.CmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, // dependencyFlags
            0, // memoryBarrierCount
            NULL, // pMemoryBarriers
            1, // bufferMemoryBarrierCount
            &buffer_memory_barrier,
            0, // imageMemoryBarrierCount
            NULL // pImageMemoryBarriers
        );

        descriptor_set =
            (descriptor_set == s->gpu_resources.descriptor_set_reduction__reduction1_to_reduction2)
            ? s->gpu_resources.descriptor_set_reduction__reduction2_to_reduction1
            : s->gpu_resources.descriptor_set_reduction__reduction1_to_reduction2;
        dst_buffer =
            (dst_buffer == s->gpu_resources.buffer_reduction_2.buffer)
            ? s->gpu_resources.buffer_reduction_1.buffer
            : s->gpu_resources.buffer_reduction_2.buffer;
    }

    // copy result to uniform buffer
    {
        const VkBuffer src_buffer = dst_buffer;
        dst_buffer = s->gpu_resources.buffer_uniforms.buffer;

        const VkBufferMemoryBarrier buffer_memory_barrier {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .srcQueueFamilyIndex = vk_ctx->queue_family_index,
            .dstQueueFamilyIndex = vk_ctx->queue_family_index,
            .buffer = src_buffer,
            .offset = 0,
            .size = sizeof(vec3),
        };
        vk_ctx->procs_dev.CmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, // dependencyFlags
            0, // memoryBarrierCount
            NULL, // pMemoryBarriers
            1, // bufferMemoryBarrierCount
            &buffer_memory_barrier,
            0, // imageMemoryBarrierCount
            NULL // pImageMemoryBarriers
        );

        const VkBufferCopy buffer_copy {
            .srcOffset = 0,
            .dstOffset = offsetof(UniformBufferData, domain_min),
            .size = sizeof(vec3),
        };
        vk_ctx->procs_dev.CmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &buffer_copy);
    }

    return PipelineBarrierSrcInfo {
        .src_stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT,
        .src_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT,
    };
};


// Use for clearing / signalling semaphores.
static void emptyQueueSubmit(
    const VulkanContext* vk_ctx,
    VkSemaphore optional_wait_semaphore,
    VkSemaphore optional_signal_semaphore,
    VkFence optional_signal_fence
) {

    VkPipelineStageFlags wait_dst_stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    VkSubmitInfo submit_info {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = optional_wait_semaphore == VK_NULL_HANDLE ? (u32)0 : (u32)1,
        .pWaitSemaphores = &optional_wait_semaphore,
        .pWaitDstStageMask = &wait_dst_stage_mask,
        .signalSemaphoreCount = optional_signal_semaphore == VK_NULL_HANDLE ? (u32)0 : (u32)1,
        .pSignalSemaphores = &optional_signal_semaphore,
    };
    VkResult result = vk_ctx->procs_dev.QueueSubmit(vk_ctx->queue, 1, &submit_info, optional_signal_fence);
    assertVk(result);
}


static void memsetZeroHostVisibleGpuBuffer(
    const VulkanContext* vk_ctx,
    const VkDeviceSize size_bytes,
    const VmaAllocation dst
) {

    ZoneScoped;

    VkResult result = VK_ERROR_UNKNOWN;

    void* p_mapped_memory = NULL;
    {
        result = vmaMapMemory(vk_ctx->vma_allocator, dst, &p_mapped_memory);
        assertVk(result);
    }

    memset(p_mapped_memory, 0, size_bytes);

    result = vmaFlushAllocation(vk_ctx->vma_allocator, dst, 0, size_bytes);
    assertVk(result);

    vmaUnmapMemory(vk_ctx->vma_allocator, dst);
};


static void uploadBufferToHostVisibleGpuMemory(
    const VulkanContext* vk_ctx,
    const VkDeviceSize size_bytes,
    const void* src,
    const VmaAllocation dst,
    const uintptr_t dst_offset
) {

    ZoneScoped;

    VkResult result = VK_ERROR_UNKNOWN;


    void* p_mapped_memory = NULL;
    {
        void* ptr = NULL;

        result = vmaMapMemory(vk_ctx->vma_allocator, dst, &ptr);
        assertVk(result);

        p_mapped_memory = (void*)( (uintptr_t)ptr + dst_offset );
    }

    memcpy(p_mapped_memory, src, size_bytes);

    result = vmaFlushAllocation(vk_ctx->vma_allocator, dst, dst_offset, size_bytes);
    assertVk(result);

    vmaUnmapMemory(vk_ctx->vma_allocator, dst);
}


// Initializes both sorted and unsorted buffers, because it's easier to not think about which one needs to
// be initialized.
static void initPositionsAndVelocitiesBuffers(
    const GpuResources* res,
    const VulkanContext* vk_ctx,
    const u32fast particle_count,
    const vec4 *const p_initial_positions
) {

    ZoneScoped;

    VkResult result = VK_ERROR_UNKNOWN;


    const VkCommandBuffer command_buffer = res->general_purpose_command_buffer;
    const VkDeviceSize data_size_bytes = particle_count * sizeof(vec4);

    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_allocation = VK_NULL_HANDLE;
    {
        const VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = data_size_bytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        const VmaAllocationCreateInfo alloc_info {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &alloc_info, &staging_buffer, &staging_allocation, NULL
        );
        assertVk(result);
    }

    uploadBufferToHostVisibleGpuMemory(vk_ctx, data_size_bytes, p_initial_positions, staging_allocation, 0);

    {
        const VkCommandBufferBeginInfo cmd_buf_begin_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        result = vk_ctx->procs_dev.BeginCommandBuffer(command_buffer, &cmd_buf_begin_info);
        assertVk(result);
        {
            const VkBufferCopy buffer_copy {
                .srcOffset = 0,
                .dstOffset = 0,
                .size = data_size_bytes,
            };
            vk_ctx->procs_dev.CmdCopyBuffer(
                command_buffer, staging_buffer, res->buffer_positions_unsorted.buffer, 1, &buffer_copy
            );
            vk_ctx->procs_dev.CmdCopyBuffer(
                command_buffer, staging_buffer, res->buffer_positions_sorted.buffer, 1, &buffer_copy
            );
        }
        result = vk_ctx->procs_dev.EndCommandBuffer(command_buffer);
        assertVk(result);
    }

    {
        const VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
        };
        result = vk_ctx->procs_dev.QueueSubmit(vk_ctx->queue, 1, &submit_info, res->fence);
        assertVk(result);

        result = vk_ctx->procs_dev.WaitForFences(vk_ctx->device, 1, &res->fence, VK_TRUE, UINT64_MAX);
        assertVk(result);

        result = vk_ctx->procs_dev.ResetFences(vk_ctx->device, 1, &res->fence);
        assertVk(result);

        result = vk_ctx->procs_dev.ResetCommandBuffer(command_buffer, 0);
        assertVk(result);
    }

    memsetZeroHostVisibleGpuBuffer(vk_ctx, data_size_bytes, staging_allocation);

    {
        const VkCommandBufferBeginInfo cmd_buf_begin_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        result = vk_ctx->procs_dev.BeginCommandBuffer(command_buffer, &cmd_buf_begin_info);
        assertVk(result);
        {
            const VkBufferCopy buffer_copy {
                .srcOffset = 0,
                .dstOffset = 0,
                .size = data_size_bytes,
            };
            vk_ctx->procs_dev.CmdCopyBuffer(
                command_buffer, staging_buffer, res->buffer_velocities_unsorted.buffer, 1, &buffer_copy
            );
            vk_ctx->procs_dev.CmdCopyBuffer(
                command_buffer, staging_buffer, res->buffer_velocities_sorted.buffer, 1, &buffer_copy
            );
        }
        result = vk_ctx->procs_dev.EndCommandBuffer(command_buffer);
        assertVk(result);
    }

    {
        const VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
        };
        result = vk_ctx->procs_dev.QueueSubmit(vk_ctx->queue, 1, &submit_info, res->fence);
        assertVk(result);

        result = vk_ctx->procs_dev.WaitForFences(vk_ctx->device, 1, &res->fence, VK_TRUE, UINT64_MAX);
        assertVk(result);

        result = vk_ctx->procs_dev.ResetFences(vk_ctx->device, 1, &res->fence);
        assertVk(result);

        result = vk_ctx->procs_dev.ResetCommandBuffer(command_buffer, 0);
        assertVk(result);
    }

    vmaDestroyBuffer(vk_ctx->vma_allocator, staging_buffer, staging_allocation);
}

static void initGpuBuffers(
    const GpuResources* res,
    const VulkanContext* vk_ctx,
    const SimData::Params *const sim_params,
    const u32 particle_count,
    const vec4 *const p_initial_positions,
    const u32 hash_modulus
) {

    ZoneScoped;

    VkResult result = VK_ERROR_UNKNOWN;


    vec3 domain_min = vec3(INFINITY);
    {
        vec3 domain_max = vec3(-INFINITY);
        for (u32fast i = 0; i < particle_count; i++)
        {
            const vec3 pos = vec3(p_initial_positions[i]);
            domain_min = glm::min(domain_min, pos);
            domain_max = glm::max(domain_max, pos);
        }
        const uvec3 cell_count = uvec3(
            glm::ceil((domain_max - domain_min) * sim_params->cell_size_reciprocal)
            + 0.5f
        );
        // 32-bit Morton codes can handle at most a 1024x1024x1024 grid.
        assert(cell_count.x < 1024 and cell_count.y < 1024 and cell_count.z < 1024);
        (void)cell_count; // to prevent "unused variable" complaints when compiling with NDEBUG
    }

    const UniformBufferData uniform_data {
        .domain_min = domain_min,
        .updated_by_host {
            .rest_particle_density = sim_params->rest_particle_density,
            .particle_interaction_radius = sim_params->particle_interaction_radius,
            .spring_rest_length = sim_params->spring_rest_length,
            .spring_stiffness = sim_params->spring_stiffness,
            .cell_size_reciprocal = sim_params->cell_size_reciprocal,
            .particle_count = particle_count,
            .hash_modulus = hash_modulus
        },
    };
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        sizeof(uniform_data),
        &uniform_data,
        res->buffer_uniforms.allocation,
        0 // dst_offset
    );

    {
        u32* p_mapped_morton_codes = NULL;
        {
            void* ptr = NULL;

            result = vmaMapMemory(
                vk_ctx->vma_allocator, res->buffer_morton_codes_or_permutation.allocation, &ptr
            );
            assertVk(result);

            p_mapped_morton_codes = (u32*)ptr;
        }

        for (u32fast i = 0; i < particle_count; i++)
        {
            vec3 pos = vec3(p_initial_positions[i]);
            u32 morton_code = cellMortonCode(cellIndex(pos, domain_min, sim_params->cell_size_reciprocal));
            p_mapped_morton_codes[i] = morton_code;
        }

        result = vmaFlushAllocation(
            vk_ctx->vma_allocator,
            res->buffer_morton_codes_or_permutation.allocation,
            0, // offset
            VK_WHOLE_SIZE
        );
        assertVk(result);

        vmaUnmapMemory(vk_ctx->vma_allocator, res->buffer_morton_codes_or_permutation.allocation);
    }

    initPositionsAndVelocitiesBuffers(res, vk_ctx, particle_count, p_initial_positions);
}


static void createBuffers(
    GpuResources* res,
    const VulkanContext* vk_ctx,
    const u32fast particle_count,
    const u32fast hash_modulus
) {

    ZoneScoped;

    VkResult result = VK_ERROR_UNKNOWN;


    struct BufferCreateInfo {
        GpuBuffer* p_buffer_out;
        VkDeviceSize size;
        VkBufferUsageFlags buffer_usage;
        VmaAllocationCreateFlags alloc_flags;
        VmaMemoryUsage mem_usage;
        VkMemoryPropertyFlags required_mem_flags;
    };

    const BufferCreateInfo buffer_infos[] {
        {
            .p_buffer_out = &res->buffer_uniforms,
            .size = sizeof(UniformBufferData),
            .buffer_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
            .mem_usage = VMA_MEMORY_USAGE_AUTO,
            .required_mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        },
        {
            .p_buffer_out = &res->buffer_positions_sorted,
            .size = particle_count * sizeof(vec4),
            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT // OPTIMIZE maybe unnecessary
                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT, // OPTIMIZE maybe unnecessary
            .alloc_flags = 0,
            .mem_usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .required_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        },
        {
            .p_buffer_out = &res->buffer_velocities_sorted,
            .size = particle_count * sizeof(vec4),
            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .alloc_flags = 0,
            .mem_usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .required_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        },
        {
            .p_buffer_out = &res->buffer_positions_unsorted,
            .size = particle_count * sizeof(vec4),
            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .alloc_flags = 0,
            .mem_usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .required_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        },
        {
            .p_buffer_out = &res->buffer_velocities_unsorted,
            .size = particle_count * sizeof(vec4),
            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .alloc_flags = 0,
            .mem_usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .required_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        },
        {
            .p_buffer_out = &res->buffer_C_begin,
            .size = (particle_count + 1) * sizeof(u32),
            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            // OPTIMIZE we should definitely try using PREFER_DEVICE + DEVICE_LOCAL + a staging buffer for
            //     this one, because the shaders access it many times per frame.
            .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .mem_usage = VMA_MEMORY_USAGE_AUTO,
            .required_mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        },
        {
            .p_buffer_out = &res->buffer_C_length,
            .size = (particle_count + 1) * sizeof(u32),
            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            // OPTIMIZE we should definitely try using PREFER_DEVICE + DEVICE_LOCAL + a staging buffer for
            //     this one, because the shaders access it many times per frame.
            .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .mem_usage = VMA_MEMORY_USAGE_AUTO,
            .required_mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        },
        {
            .p_buffer_out = &res->buffer_H_begin,
            .size = hash_modulus * sizeof(u32),
            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            // OPTIMIZE we should definitely try using PREFER_DEVICE + DEVICE_LOCAL + a staging buffer for
            //     this one, because the shaders access it many times per frame.
            .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .mem_usage = VMA_MEMORY_USAGE_AUTO,
            .required_mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        },
        {
            .p_buffer_out = &res->buffer_H_length,
            .size = hash_modulus * sizeof(u32),
            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            // OPTIMIZE we should definitely try using PREFER_DEVICE + DEVICE_LOCAL + a staging buffer for
            //     this one, because the shaders access it many times per frame.
            .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .mem_usage = VMA_MEMORY_USAGE_AUTO,
            .required_mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        },
        {
            .p_buffer_out = &res->buffer_reduction_1,
            .size = particle_count * sizeof(vec4), // OPTIMIZE this can be smaller (divCeil(size, 2)?)
            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .alloc_flags = 0,
            .mem_usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .required_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        },
        {
            .p_buffer_out = &res->buffer_reduction_2,
            .size = particle_count * sizeof(vec4), // OPTIMIZE this can be smaller (divCeil(size, 4)?)
            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                          | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .alloc_flags = 0,
            .mem_usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .required_mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        },
        {
            .p_buffer_out = &res->buffer_morton_codes_or_permutation,
            .size = particle_count * sizeof(u32),
            .buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .alloc_flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
            .mem_usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            .required_mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        },
    };
    const u32fast buffer_info_count = ARRAY_SIZE(buffer_infos);

    for (u32fast i = 0; i < buffer_info_count; i++)
    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = buffer_infos[i].size,
            .usage = buffer_infos[i].buffer_usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo alloc_info {
            .flags = buffer_infos[i].alloc_flags,
            .usage = buffer_infos[i].mem_usage,
            .requiredFlags = buffer_infos[i].required_mem_flags,
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &alloc_info,
            &buffer_infos[i].p_buffer_out->buffer,
            &buffer_infos[i].p_buffer_out->allocation,
            &buffer_infos[i].p_buffer_out->allocation_info
        );
        assertVk(result);
    }
}


static void createDescriptorStuff(
    GpuResources* res,
    const VulkanContext* vk_ctx
) {
    using descriptor_management::DescriptorSetLayout;

    VkDescriptorSetLayoutBinding layout_bindings_general[LAYOUT_BINDING_COUNT__GENERAL] {};
    for (u32 i = 0; i < LAYOUT_BINDING_COUNT__GENERAL; i++)
    {
        layout_bindings_general[i] = VkDescriptorSetLayoutBinding {
            .binding = i,
            .descriptorType = DESCRIPTOR_SET_LAYOUT__GENERAL[i],
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }

    VkDescriptorSetLayoutBinding layout_bindings_reduction[LAYOUT_BINDING_COUNT__REDUCTION] {};
    for (u32 i = 0; i < LAYOUT_BINDING_COUNT__REDUCTION; i++)
    {
        layout_bindings_reduction[i] = VkDescriptorSetLayoutBinding {
            .binding = i,
            .descriptorType = DESCRIPTOR_SET_LAYOUT__REDUCTION[i],
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }

    constexpr u32fast layout_count = 2;
    const DescriptorSetLayout layout_infos[layout_count] {
        DescriptorSetLayout {
            .binding_count = LAYOUT_BINDING_COUNT__GENERAL,
            .p_bindings = layout_bindings_general,
        },
        DescriptorSetLayout {
            .binding_count = LAYOUT_BINDING_COUNT__REDUCTION,
            .p_bindings = layout_bindings_reduction,
        },
    };

    const u32 descriptor_set_counts[layout_count] { 1, 3 };
    constexpr u32 total_descriptor_set_count = 4;

    VkDescriptorSetLayout descriptor_set_layouts[layout_count] {};
    VkDescriptorSet descriptor_sets[total_descriptor_set_count] {};
    {
        descriptor_management::createDescriptorPoolAndSets(
            vk_ctx,
            layout_count,
            layout_infos,
            descriptor_set_counts,
            &res->descriptor_pool,
            descriptor_set_layouts,
            descriptor_sets
        );
    }

    res->descriptor_set_layout_main = descriptor_set_layouts[0];
    res->descriptor_set_layout_reduction = descriptor_set_layouts[1];

    res->descriptor_set_main = descriptor_sets[0];
    res->descriptor_set_reduction__positions_to_reduction1 = descriptor_sets[1];
    res->descriptor_set_reduction__reduction1_to_reduction2 = descriptor_sets[2];
    res->descriptor_set_reduction__reduction2_to_reduction1 = descriptor_sets[3];

    // initialize descriptors --------------------------------------------------------------------------------

    {
        ZoneScopedN("WriteDescriptorSets");

        const VkDescriptorBufferInfo buffer_infos[LAYOUT_BINDING_COUNT__GENERAL] {
            [LAYOUT_BINDING_GENERAL__UNIFORMS] = { .buffer = res->buffer_uniforms.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            [LAYOUT_BINDING_GENERAL__POSITIONS_SORTED] = { .buffer = res->buffer_positions_sorted.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            [LAYOUT_BINDING_GENERAL__VELOCITIES_SORTED] = { .buffer = res->buffer_velocities_sorted.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            [LAYOUT_BINDING_GENERAL__POSITIONS_UNSORTED] = { .buffer = res->buffer_positions_unsorted.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            [LAYOUT_BINDING_GENERAL__VELOCITIES_UNSORTED] = { .buffer = res->buffer_velocities_unsorted.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            [LAYOUT_BINDING_GENERAL__C_BEGIN] = { .buffer = res->buffer_C_begin.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            [LAYOUT_BINDING_GENERAL__C_LENGTH] = { .buffer = res->buffer_C_length.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            [LAYOUT_BINDING_GENERAL__H_BEGIN] = { .buffer = res->buffer_H_begin.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            [LAYOUT_BINDING_GENERAL__H_LENGTH] = { .buffer = res->buffer_H_length.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            [LAYOUT_BINDING_GENERAL__MORTON_CODES_OR_PERMUTATION] = { .buffer = res->buffer_morton_codes_or_permutation.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
        };

        VkWriteDescriptorSet writes[LAYOUT_BINDING_COUNT__GENERAL] {};
        {
            for (u32 binding_idx = 0; binding_idx < LAYOUT_BINDING_COUNT__GENERAL; binding_idx++)
            {
                writes[binding_idx] = VkWriteDescriptorSet {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = res->descriptor_set_main,
                    .dstBinding = binding_idx,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = DESCRIPTOR_SET_LAYOUT__GENERAL[binding_idx],
                    .pBufferInfo = &buffer_infos[binding_idx],
                };
            }
        }

        vk_ctx->procs_dev.UpdateDescriptorSets(vk_ctx->device, LAYOUT_BINDING_COUNT__GENERAL, writes, 0, NULL);
    }

    {
        VkDescriptorBufferInfo buffer_info_positions
            { .buffer = res->buffer_positions_unsorted.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
        VkDescriptorBufferInfo buffer_info_reduction1
            { .buffer = res->buffer_reduction_1.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
        VkDescriptorBufferInfo buffer_info_reduction2
            { .buffer = res->buffer_reduction_2.buffer, .offset = 0, .range = VK_WHOLE_SIZE };

        {
            const VkWriteDescriptorSet write_template = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = VK_NULL_HANDLE, // to be filled
                .dstBinding = UINT32_MAX, // to be filled
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = NULL, // to be filled
            };

            constexpr u32 write_count = 6;
            VkWriteDescriptorSet writes[write_count] {};

            writes[0] = write_template;
            writes[0].dstSet = res->descriptor_set_reduction__positions_to_reduction1;
            writes[0].dstBinding = 0;
            writes[0].pBufferInfo = &buffer_info_positions;
            writes[1] = write_template;
            writes[1].dstSet = res->descriptor_set_reduction__positions_to_reduction1;
            writes[1].dstBinding = 1;
            writes[1].pBufferInfo = &buffer_info_reduction1;

            writes[2] = write_template;
            writes[2].dstSet = res->descriptor_set_reduction__reduction1_to_reduction2;
            writes[2].dstBinding = 0;
            writes[2].pBufferInfo = &buffer_info_reduction1;
            writes[3] = write_template;
            writes[3].dstSet = res->descriptor_set_reduction__reduction1_to_reduction2;
            writes[3].dstBinding = 1;
            writes[3].pBufferInfo = &buffer_info_reduction2;

            writes[4] = write_template;
            writes[4].dstSet = res->descriptor_set_reduction__reduction2_to_reduction1;
            writes[4].dstBinding = 0;
            writes[4].pBufferInfo = &buffer_info_reduction2;
            writes[5] = write_template;
            writes[5].dstSet = res->descriptor_set_reduction__reduction2_to_reduction1;
            writes[5].dstBinding = 1;
            writes[5].pBufferInfo = &buffer_info_reduction1;

            vk_ctx->procs_dev.UpdateDescriptorSets(vk_ctx->device, write_count, writes, 0, NULL);
        }
    }
};


static void createComputePipeline_updateParticles(
    const VulkanContext* vk_ctx,
    const u32 workgroup_size,
    const VkDescriptorSetLayout descriptor_set_layout,
    VkPipeline* pipeline_out,
    VkPipelineLayout* pipeline_layout_out
) {

    ZoneScoped;

    VkResult result = VK_ERROR_UNKNOWN;


    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    {
        VkPushConstantRange push_constant_range {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(ParticleUpdatePushConstants),
        };

        VkPipelineLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range,
        };
        result = vk_ctx->procs_dev.CreatePipelineLayout(vk_ctx->device, &layout_info, NULL, &pipeline_layout);
        assertVk(result);
    }
    *pipeline_layout_out = pipeline_layout;

    const VkSpecializationMapEntry specialization_map_entry {
        .constantID = COMPUTE_SHADER_SPECIALIZATION_CONSTANT_IDS.local_size_x,
        .offset = 0,
        .size = sizeof(u32),
    };

    const VkSpecializationInfo specialization_info {
        .mapEntryCount = 1,
        .pMapEntries = &specialization_map_entry,
        .dataSize = sizeof(u32),
        .pData = &workgroup_size,
    };

    VkShaderModule shader_module = VK_NULL_HANDLE;
    {
        size_t spirv_size = 0;
        // TODO FIXME Create some central shader/pipeline manager so that we can hot-reload this without
        // rewriting all the hot-reloading code that's currently in graphics.cpp.
        void* p_spirv = file_util::readEntireFile("build/shaders/fluidSim_updateParticles.comp.spv", &spirv_size);
        alwaysAssert(p_spirv != NULL);
        alwaysAssert(spirv_size != 0);
        defer(free(p_spirv));

        alwaysAssert((uintptr_t)p_spirv % alignof(u32) == 0);
        alwaysAssert(spirv_size % sizeof(u32) == 0);

        const VkShaderModuleCreateInfo shader_module_info {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = (u32*)p_spirv,
        };
        result = vk_ctx->procs_dev.CreateShaderModule(vk_ctx->device, &shader_module_info, NULL, &shader_module);
        assertVk(result);
    }
    defer(vk_ctx->procs_dev.DestroyShaderModule(vk_ctx->device, shader_module, NULL));

    const VkPipelineShaderStageCreateInfo shader_stage_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = &specialization_info,
    };

    const VkComputePipelineCreateInfo pipeline_info {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = shader_stage_info,
        .layout = pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    result = vk_ctx->procs_dev.CreateComputePipelines(
        vk_ctx->device,
        VK_NULL_HANDLE, // pipelineCache
        1, // createInfoCount
        &pipeline_info,
        NULL, // pAllocator
        pipeline_out
    );
    assertVk(result);
}

static void createComputePipeline_sortParticles(
    const VulkanContext* vk_ctx,
    const u32 workgroup_size,
    const VkDescriptorSetLayout descriptor_set_layout,
    VkPipeline* pipeline_out,
    VkPipelineLayout* pipeline_layout_out
) {

    ZoneScoped;

    VkResult result = VK_ERROR_UNKNOWN;


    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    {
        VkPipelineLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = NULL,
        };
        result = vk_ctx->procs_dev.CreatePipelineLayout(vk_ctx->device, &layout_info, NULL, &pipeline_layout);
        assertVk(result);
    }
    *pipeline_layout_out = pipeline_layout;

    const VkSpecializationMapEntry specialization_map_entry {
        .constantID = COMPUTE_SHADER_SPECIALIZATION_CONSTANT_IDS.local_size_x,
        .offset = 0,
        .size = sizeof(u32),
    };

    const VkSpecializationInfo specialization_info {
        .mapEntryCount = 1,
        .pMapEntries = &specialization_map_entry,
        .dataSize = sizeof(u32),
        .pData = &workgroup_size,
    };

    VkShaderModule shader_module = VK_NULL_HANDLE;
    {
        size_t spirv_size = 0;
        // TODO FIXME Create some central shader/pipeline manager so that we can hot-reload this without
        // rewriting all the hot-reloading code that's currently in graphics.cpp.
        void* p_spirv = file_util::readEntireFile("build/shaders/fluidSim_sortParticles.comp.spv", &spirv_size);
        alwaysAssert(p_spirv != NULL);
        alwaysAssert(spirv_size != 0);
        defer(free(p_spirv));

        alwaysAssert((uintptr_t)p_spirv % alignof(u32) == 0);
        alwaysAssert(spirv_size % sizeof(u32) == 0);

        const VkShaderModuleCreateInfo shader_module_info {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = (u32*)p_spirv,
        };
        result = vk_ctx->procs_dev.CreateShaderModule(vk_ctx->device, &shader_module_info, NULL, &shader_module);
        assertVk(result);
    }
    defer(vk_ctx->procs_dev.DestroyShaderModule(vk_ctx->device, shader_module, NULL));

    const VkPipelineShaderStageCreateInfo shader_stage_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = &specialization_info,
    };

    const VkComputePipelineCreateInfo pipeline_info {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = shader_stage_info,
        .layout = pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    result = vk_ctx->procs_dev.CreateComputePipelines(
        vk_ctx->device,
        VK_NULL_HANDLE, // pipelineCache
        1, // createInfoCount
        &pipeline_info,
        NULL, // pAllocator
        pipeline_out
    );
    assertVk(result);
}

static void createComputePipeline_computeMortonCodes(
    const VulkanContext* vk_ctx,
    const u32 workgroup_size,
    const VkDescriptorSetLayout descriptor_set_layout,
    VkPipeline* pipeline_out,
    VkPipelineLayout* pipeline_layout_out
) {

    ZoneScoped;

    VkResult result = VK_ERROR_UNKNOWN;


    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    {
        VkPipelineLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = NULL,
        };
        result = vk_ctx->procs_dev.CreatePipelineLayout(vk_ctx->device, &layout_info, NULL, &pipeline_layout);
        assertVk(result);
    }
    *pipeline_layout_out = pipeline_layout;

    const VkSpecializationMapEntry specialization_map_entry {
        .constantID = COMPUTE_SHADER_SPECIALIZATION_CONSTANT_IDS.local_size_x,
        .offset = 0,
        .size = sizeof(u32),
    };

    const VkSpecializationInfo specialization_info {
        .mapEntryCount = 1,
        .pMapEntries = &specialization_map_entry,
        .dataSize = sizeof(u32),
        .pData = &workgroup_size,
    };

    VkShaderModule shader_module = VK_NULL_HANDLE;
    {
        size_t spirv_size = 0;
        // TODO FIXME Create some central shader/pipeline manager so that we can hot-reload this without
        // rewriting all the hot-reloading code that's currently in graphics.cpp.
        void* p_spirv = file_util::readEntireFile("build/shaders/fluidSim_computeMortonCodes.comp.spv", &spirv_size);
        alwaysAssert(p_spirv != NULL);
        alwaysAssert(spirv_size != 0);
        defer(free(p_spirv));

        alwaysAssert((uintptr_t)p_spirv % alignof(u32) == 0);
        alwaysAssert(spirv_size % sizeof(u32) == 0);

        const VkShaderModuleCreateInfo shader_module_info {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = (u32*)p_spirv,
        };
        result = vk_ctx->procs_dev.CreateShaderModule(vk_ctx->device, &shader_module_info, NULL, &shader_module);
        assertVk(result);
    }
    defer(vk_ctx->procs_dev.DestroyShaderModule(vk_ctx->device, shader_module, NULL));

    const VkPipelineShaderStageCreateInfo shader_stage_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = &specialization_info,
    };

    const VkComputePipelineCreateInfo pipeline_info {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = shader_stage_info,
        .layout = pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    result = vk_ctx->procs_dev.CreateComputePipelines(
        vk_ctx->device,
        VK_NULL_HANDLE, // pipelineCache
        1, // createInfoCount
        &pipeline_info,
        NULL, // pAllocator
        pipeline_out
    );
    assertVk(result);
}

static void createComputePipeline_computeMin(
    const VulkanContext* vk_ctx,
    const u32 workgroup_size,
    const VkDescriptorSetLayout descriptor_set_layout,
    VkPipeline* pipeline_out,
    VkPipelineLayout* pipeline_layout_out
) {

    ZoneScoped;

    VkResult result = VK_ERROR_UNKNOWN;


    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    {
        VkPushConstantRange push_constant_range {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(ReductionPushConstants),
        };

        VkPipelineLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range,
        };
        result = vk_ctx->procs_dev.CreatePipelineLayout(vk_ctx->device, &layout_info, NULL, &pipeline_layout);
        assertVk(result);
    }
    *pipeline_layout_out = pipeline_layout;

    const VkSpecializationMapEntry specialization_map_entry {
        .constantID = COMPUTE_SHADER_SPECIALIZATION_CONSTANT_IDS.local_size_x,
        .offset = 0,
        .size = sizeof(u32),
    };

    const VkSpecializationInfo specialization_info {
        .mapEntryCount = 1,
        .pMapEntries = &specialization_map_entry,
        .dataSize = sizeof(u32),
        .pData = &workgroup_size,
    };

    VkShaderModule shader_module = VK_NULL_HANDLE;
    {
        size_t spirv_size = 0;
        // TODO FIXME Create some central shader/pipeline manager so that we can hot-reload this without
        // rewriting all the hot-reloading code that's currently in graphics.cpp.
        void* p_spirv = file_util::readEntireFile("build/shaders/fluidSim_computeMin.comp.spv", &spirv_size);
        alwaysAssert(p_spirv != NULL);
        alwaysAssert(spirv_size != 0);
        defer(free(p_spirv));

        alwaysAssert((uintptr_t)p_spirv % alignof(u32) == 0);
        alwaysAssert(spirv_size % sizeof(u32) == 0);

        const VkShaderModuleCreateInfo shader_module_info {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv_size,
            .pCode = (u32*)p_spirv,
        };
        result = vk_ctx->procs_dev.CreateShaderModule(vk_ctx->device, &shader_module_info, NULL, &shader_module);
        assertVk(result);
    }
    defer(vk_ctx->procs_dev.DestroyShaderModule(vk_ctx->device, shader_module, NULL));

    const VkPipelineShaderStageCreateInfo shader_stage_info {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = &specialization_info,
    };

    const VkComputePipelineCreateInfo pipeline_info {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = shader_stage_info,
        .layout = pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    result = vk_ctx->procs_dev.CreateComputePipelines(
        vk_ctx->device,
        VK_NULL_HANDLE, // pipelineCache
        1, // createInfoCount
        &pipeline_info,
        NULL, // pAllocator
        pipeline_out
    );
    assertVk(result);
}

static void createComputePipelines(
    GpuResources* res,
    const VulkanContext* vk_ctx,
    const u32 workgroup_size
) {

    ZoneScoped;

    assert(res->descriptor_set_layout_main != VK_NULL_HANDLE);
    assert(res->descriptor_set_layout_reduction != VK_NULL_HANDLE);

    createComputePipeline_computeMin(
        vk_ctx, workgroup_size, res->descriptor_set_layout_reduction,
        &res->pipeline_computeMin, &res->pipeline_layout_computeMin
    );
    createComputePipeline_computeMortonCodes(
        vk_ctx, workgroup_size, res->descriptor_set_layout_main,
        &res->pipeline_computeMortonCodes, &res->pipeline_layout_computeMortonCodes
    );
    createComputePipeline_updateParticles(
        vk_ctx, workgroup_size, res->descriptor_set_layout_main,
        &res->pipeline_updateParticles, &res->pipeline_layout_updateParticles
    );
    createComputePipeline_sortParticles(
        vk_ctx, workgroup_size, res->descriptor_set_layout_main,
        &res->pipeline_sortParticles, &res->pipeline_layout_sortParticles
    );
};


extern "C" void setParams(SimData* s, const SimParameters* params) {
    s->parameters.rest_particle_density = params->rest_particle_density;
    s->parameters.spring_stiffness = params->spring_stiffness;

    // Number of particles contained in sphere at rest ~= sphere volume * rest particle density.
    // :: N = (4/3 pi r^3) rho
    // :: r = cuberoot(N * 3 / (4 pi rho)).
    s->parameters.particle_interaction_radius = cbrtf(
        (f32)params->rest_particle_interaction_count_approx * 3.f / (4.f * PI * params->rest_particle_density)
    );

    // TODO FIXME didn't really think about a good way to compute this
    s->parameters.spring_rest_length = s->parameters.particle_interaction_radius * 0.5f;

    const f32 cell_size = s->parameters.particle_interaction_radius;
    s->parameters.cell_size = cell_size;
    s->parameters.cell_size_reciprocal = 1.0f / cell_size;

    LOG_F(INFO, "Set fluid sim parameters: "
        "REST_PARTICLE_DENSITY = %f, "
        "SPRING_STIFFNESS = %f, "
        "SPRING_REST_LENGTH = %f, "
        "PARTICLE_INTERACTION_RADIUS = %f, "
        "CELL_SIZE = %f.",
        s->parameters.rest_particle_density,
        s->parameters.spring_stiffness,
        s->parameters.spring_rest_length,
        s->parameters.particle_interaction_radius,
        s->parameters.cell_size
    );
}


static void sortCells(

    thread_pool::ThreadPool* thread_pool,
    u32 thread_count,

    const u32fast cell_count,

    u32 **const pp_cells,
    u32 **const pp_lengths,
    u32 **const pp_cells_scratch,
    u32 **const pp_lengths_scratch,

    KeyVal *const p_scratch1,
    KeyVal *const p_scratch2,

    const u32 *const p_morton_codes,

    const u32 hash_modulus
) {

    ZoneScoped;

    KeyVal* p_keyvals = p_scratch1;
    KeyVal* p_scratch = p_scratch2;

    u32* p_cells_in = *pp_cells;
    u32* p_lengths_in = *pp_lengths;
    u32* p_cells_out = *pp_cells_scratch;
    u32* p_lengths_out = *pp_lengths_scratch;

    {
        ZoneScopedN("init cell hashes");

        for (u32 i = 0; i < cell_count; i++)
        {
            const u32 particle_idx = p_cells_in[i];
            const u32 morton_code = p_morton_codes[particle_idx];
            const u32 cell_hash = mortonCodeHash(morton_code, hash_modulus);

            p_keyvals[i].key = cell_hash;
            p_keyvals[i].val = i;
        }
    }

    alwaysAssert(thread_count > 0);
    mergeSortMultiThreaded(
        thread_pool,
        thread_count,
        cell_count,
        p_keyvals,
        p_scratch
    );

    {
        ZoneScopedN("sort by permutation");

        for (u32fast i = 0; i < cell_count; i++)
        {
            u32fast src_idx = p_keyvals[i].val;

            p_cells_out[i] = p_cells_in[src_idx];
            p_cells_out[i] = p_cells_in[src_idx];

            p_lengths_out[i] = p_lengths_in[src_idx];
            p_lengths_out[i] = p_lengths_in[src_idx];
        }
    }

    *pp_cells = p_cells_out;
    *pp_lengths = p_lengths_out;
    *pp_cells_scratch = p_cells_in;
    *pp_lengths_scratch = p_lengths_in;
};


static u32fast getNextPrimeNumberExclusive(u32fast n) {

    ZoneScoped;

    if (n == 0 || n == 1) return 2;

    n++; // never return input
    if (!(n & 1)) n++; // if even, make it odd

    // maybe this is 1 too large, that's fine
    const u32fast max = (u32fast)(ceil(sqrt((f64)n)) + 0.5);

    while (true)
    {
        bool prime = true;
        for (u32fast i = 3; i <= max; i++)
        {
            if (n % i == 0)
            {
                prime = false;
                break;
            }
        }
        if (prime) return n;
        n += 2;
    }
}


static GpuResources createGpuResources(
    const VulkanContext* vk_ctx,
    u32fast particle_count,
    u32fast hash_modulus
) {

    ZoneScoped;

    VkResult result = VK_ERROR_UNKNOWN;
    GpuResources resources {};


    u32 workgroup_size = 0;
    u32 workgroup_count = 0;
    {
        const u32 max_workgroup_size = vk_ctx->physical_device_properties.limits.maxComputeWorkGroupSize[0];
        const u32 max_workgroup_count = vk_ctx->physical_device_properties.limits.maxComputeWorkGroupCount[0];

        workgroup_size = 128; // OPTIMIZE find a decent way to tune this number
        workgroup_size = glm::min(workgroup_size, max_workgroup_size);

        workgroup_count = divCeil((u32)particle_count, workgroup_size);
        if (workgroup_count > max_workgroup_count)
        {
            workgroup_count = max_workgroup_count;
            workgroup_size = divCeil((u32)particle_count, workgroup_count);
        }

        alwaysAssert(workgroup_size > 0);
        alwaysAssert(workgroup_count > 0);
        alwaysAssert(workgroup_size <= max_workgroup_size);
        alwaysAssert(workgroup_count <= max_workgroup_count);

        assert(workgroup_size * workgroup_count >= particle_count);

        resources.workgroup_size = workgroup_size;
        resources.workgroup_count = workgroup_count;
    }


    {
        VkCommandPoolCreateInfo pool_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = vk_ctx->queue_family_index
        };
        result = vk_ctx->procs_dev.CreateCommandPool(vk_ctx->device, &pool_info, NULL, &resources.command_pool);
        assertVk(result);
    }
    {
        constexpr u32 command_buffer_count = 2;
        VkCommandBuffer command_buffers[command_buffer_count] {};

        VkCommandBufferAllocateInfo alloc_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = resources.command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = command_buffer_count,
        };
        result = vk_ctx->procs_dev.AllocateCommandBuffers(vk_ctx->device, &alloc_info, command_buffers);
        assertVk(result);

        resources.general_purpose_command_buffer = command_buffers[0];
        resources.morton_code_command_buffer = command_buffers[1];
    }

    {
        const VkSemaphoreCreateInfo semaphore_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        result = vk_ctx->procs_dev.CreateSemaphore(
            vk_ctx->device, &semaphore_info, NULL, &resources.particle_update_finished_semaphore
        );

        const VkFenceCreateInfo fence_info { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        result = vk_ctx->procs_dev.CreateFence(vk_ctx->device, &fence_info, NULL, &resources.fence);
        assertVk(result);
    }


    createBuffers(&resources, vk_ctx, particle_count, hash_modulus);
    createDescriptorStuff(&resources, vk_ctx);
    createComputePipelines(&resources, vk_ctx, workgroup_size);


    return resources;
}


static void destroyGpuResources(GpuResources* res, const VulkanContext* vk_ctx) {

    ZoneScoped;

    VkResult result = vk_ctx->procs_dev.QueueWaitIdle(vk_ctx->queue);
    assertVk(result);

    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_uniforms.buffer, res->buffer_uniforms.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_positions_sorted.buffer, res->buffer_positions_sorted.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_velocities_sorted.buffer, res->buffer_velocities_sorted.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_positions_unsorted.buffer, res->buffer_positions_unsorted.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_velocities_unsorted.buffer, res->buffer_velocities_unsorted.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_C_begin.buffer, res->buffer_C_begin.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_C_length.buffer, res->buffer_C_length.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_H_begin.buffer, res->buffer_H_begin.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_H_length.buffer, res->buffer_H_length.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_reduction_1.buffer, res->buffer_reduction_1.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_reduction_2.buffer, res->buffer_reduction_2.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_morton_codes_or_permutation.buffer, res->buffer_morton_codes_or_permutation.allocation);

    vk_ctx->procs_dev.DestroyCommandPool(vk_ctx->device, res->command_pool, NULL);

    vk_ctx->procs_dev.DestroyDescriptorSetLayout(vk_ctx->device, res->descriptor_set_layout_main, NULL);
    vk_ctx->procs_dev.DestroyDescriptorSetLayout(vk_ctx->device, res->descriptor_set_layout_reduction, NULL);
    vk_ctx->procs_dev.DestroyDescriptorPool(vk_ctx->device, res->descriptor_pool, NULL);


    vk_ctx->procs_dev.DestroyPipeline(vk_ctx->device, res->pipeline_updateParticles, NULL);
    vk_ctx->procs_dev.DestroyPipelineLayout(vk_ctx->device, res->pipeline_layout_updateParticles, NULL);

    vk_ctx->procs_dev.DestroyPipeline(vk_ctx->device, res->pipeline_computeMin, NULL);
    vk_ctx->procs_dev.DestroyPipelineLayout(vk_ctx->device, res->pipeline_layout_computeMin, NULL);

    vk_ctx->procs_dev.DestroyPipeline(vk_ctx->device, res->pipeline_computeMortonCodes, NULL);
    vk_ctx->procs_dev.DestroyPipelineLayout(vk_ctx->device, res->pipeline_layout_computeMortonCodes, NULL);

    vk_ctx->procs_dev.DestroyPipeline(vk_ctx->device, res->pipeline_sortParticles, NULL);
    vk_ctx->procs_dev.DestroyPipelineLayout(vk_ctx->device, res->pipeline_layout_sortParticles, NULL);


    vk_ctx->procs_dev.DestroySemaphore(vk_ctx->device, res->particle_update_finished_semaphore, NULL);
    vk_ctx->procs_dev.DestroyFence(vk_ctx->device, res->fence, NULL);
}


static void downloadBufferFromHostVisibleGpuMemory(
    const VulkanContext* vk_ctx,
    const u32fast size_bytes,
    const VmaAllocation src,
    void* dst,
    const VkDeviceSize src_offset
) {

    VkResult result = VK_ERROR_UNKNOWN;


    const void* p_mapped_memory = NULL;
    {
        void* ptr = NULL;

        result = vmaMapMemory(vk_ctx->vma_allocator, src, &ptr);
        assertVk(result);

        p_mapped_memory = (void*)( (uintptr_t)ptr + (uintptr_t)src_offset );
    }

    result = vmaInvalidateAllocation(vk_ctx->vma_allocator, src, src_offset, size_bytes);
    assertVk(result);

    {
        ZoneScopedN("memcpy");
        memcpy(dst, p_mapped_memory, size_bytes);
    }

    vmaUnmapMemory(vk_ctx->vma_allocator, src);
}


static void uploadDataToGpu(const SimData* s, const VulkanContext* vk_ctx) {

    ZoneScoped;

    {
        const UniformBufferData::UpdatedByHost uniform_data {
            .rest_particle_density = s->parameters.rest_particle_density,
            .particle_interaction_radius = s->parameters.particle_interaction_radius,
            .spring_rest_length = s->parameters.spring_rest_length,
            .spring_stiffness = s->parameters.spring_stiffness,
            .cell_size_reciprocal = s->parameters.cell_size_reciprocal,

            .particle_count = (u32)s->particle_count,
            .hash_modulus = s->hash_modulus,
        };
        uploadBufferToHostVisibleGpuMemory(
            vk_ctx,
            sizeof(uniform_data),
            &uniform_data,
            s->gpu_resources.buffer_uniforms.allocation,
            offsetof(UniformBufferData, updated_by_host)
        );
    }

    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->particle_count * sizeof(u32),
        s->p_permutation,
        s->gpu_resources.buffer_morton_codes_or_permutation.allocation,
        0 // dst_offset
    );

    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->cell_count * sizeof(*s->C_begin),
        s->C_begin,
        s->gpu_resources.buffer_C_begin.allocation,
        0 // dst_offset
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->cell_count * sizeof(*s->C_length),
        s->C_length,
        s->gpu_resources.buffer_C_length.allocation,
        0 // dst_offset
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->hash_modulus * sizeof(*s->H_begin),
        s->H_begin,
        s->gpu_resources.buffer_H_begin.allocation,
        0 // dst_offset
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->hash_modulus * sizeof(*s->H_length),
        s->H_length,
        s->gpu_resources.buffer_H_length.allocation,
        0 // dst_offset
    );
}


static void downloadDataFromGpu(SimData* s, const VulkanContext* vk_ctx, vec3* p_domain_min_out) {

    ZoneScoped;

    // OPTIMIZE why copy the buffer? We can just map and use it directly, since it's host-allocated anyway.
    downloadBufferFromHostVisibleGpuMemory(
        vk_ctx,
        s->particle_count * sizeof(u32),
        s->gpu_resources.buffer_morton_codes_or_permutation.allocation,
        s->p_morton_codes,
        0 // src_offset
    );
    downloadBufferFromHostVisibleGpuMemory(
        vk_ctx,
        sizeof(vec3),
        s->gpu_resources.buffer_uniforms.allocation,
        p_domain_min_out,
        offsetof(UniformBufferData, domain_min)
    );
}


static void allocateHostBuffers(SimData* s, u32fast particle_count, u32fast hash_modulus) {

    ZoneScoped;

    s->p_cells_scratch_buffer1 = callocArray(particle_count + 1, u32);
    s->p_cells_scratch_buffer2 = callocArray(particle_count + 1, u32);

    s->p_scratch_keyval_buffer_1 = callocArray(particle_count + 1, KeyVal);
    s->p_scratch_keyval_buffer_2 = callocArray(particle_count + 1, KeyVal);

    s->C_begin = callocArray(particle_count + 1, u32);
    s->C_length = callocArray(particle_count + 1, u32);

    s->H_begin = callocArray(hash_modulus, u32);
    s->H_length = callocArray(hash_modulus, u32);

    s->p_morton_codes = callocArray(particle_count, u32);
    s->p_permutation = callocArray(particle_count, u32);
}


extern "C" SimData create(
    const SimParameters* params,
    const VulkanContext* vk_ctx,
    u32fast particle_count,
    const vec4* p_initial_positions
) {

    ZoneScoped;

    LOG_F(INFO, "Initializing fluid sim.");


    // smallest prime number larger than the maximum number of particles
    // OPTIMIZE profile this and optimize if too slow
    u32fast hash_modulus = getNextPrimeNumberExclusive(particle_count);
    assert(hash_modulus <= UINT32_MAX);

    SimData s {};
    {
        s.particle_count = particle_count;
        s.cell_count = 0;
        s.hash_modulus = (u32)hash_modulus;

        allocateHostBuffers(&s, particle_count, hash_modulus);
        setParams(&s, params);
        s.gpu_resources = createGpuResources(vk_ctx, particle_count, hash_modulus);

        {
            long processor_count = sysconf(_SC_NPROCESSORS_ONLN);

            if (processor_count < 0)
            {
                const int err = errno;
                const char* err_description = strerror(err);
                if (err_description == NULL) err_description = "(NO DESCRIPTION PROVIDED)";

                LOG_F(
                    ERROR, "Failed to get processor count; (errno %i, description `%s`).",
                    err, err_description
                );
                processor_count = 1;
            }
            else if (processor_count == 0)
            {
                LOG_F(ERROR, "Failed to get processor count (got 0).");
                processor_count = 1;
            }

            LOG_F(INFO, "Using processor_count=%li.", processor_count);
            s.processor_count = (u32)processor_count;
        }
    }

    initGpuBuffers(
        &s.gpu_resources,
        vk_ctx,
        &s.parameters,
        (u32)particle_count,
        p_initial_positions,
        (u32)hash_modulus
    );

    // signal the fence, so that we don't deadlock when waiting for it in `advance()`.
    emptyQueueSubmit(vk_ctx, VK_NULL_HANDLE, VK_NULL_HANDLE, s.gpu_resources.fence);


    LOG_F(
         INFO,
         "Initialized fluid sim with %" PRIuFAST32 " particles, workgroup_size=%u, workgroup_count=%u.",
         s.particle_count, s.gpu_resources.workgroup_size, s.gpu_resources.workgroup_count
     );

    return s;
}


extern "C" void destroy(SimData* s, const VulkanContext* vk_ctx) {

    ZoneScoped;

    destroyGpuResources(&s->gpu_resources, vk_ctx);

    free(s->p_cells_scratch_buffer1);
    free(s->p_cells_scratch_buffer2);

    free(s->p_scratch_keyval_buffer_1);
    free(s->p_scratch_keyval_buffer_2);

    free(s->C_begin);
    free(s->C_length);

    free(s->H_begin);
    free(s->H_length);

    free(s->p_morton_codes);
    free(s->p_permutation);

    memset(s, 0, sizeof(*s));
}


extern "C" void advance(
    SimData* s,
    const VulkanContext* vk_ctx,
    thread_pool::ThreadPool* thread_pool,
    f32 delta_t,
    VkSemaphore optional_wait_semaphore,
    VkSemaphore particle_update_finished_signal_semaphore_optional
) {

    ZoneScoped;

    VkResult result = VK_ERROR_UNKNOWN;

    assert(delta_t > 1e-5); // assert nonzero


    {
        ZoneScopedN("WaitForFences");
        result = vk_ctx->procs_dev.WaitForFences(vk_ctx->device, 1, &s->gpu_resources.fence, true, UINT64_MAX);
        assertVk(result);
    }
    result = vk_ctx->procs_dev.ResetFences(vk_ctx->device, 1, &s->gpu_resources.fence);
    assertVk(result);

    vec3 domain_min = vec3(INFINITY);
    downloadDataFromGpu(s, vk_ctx, &domain_min);

    const u32fast particle_count = s->particle_count;

    // TODO FIXME WARNING:
    //     If the domain spans more than 1024 cells in any dimension, the simulation is invalid, because
    //     we store 30-bit Morton codes (so a uvec3 cell index is 10 bits per dimension; 2^10 = 1024).
    //     We should assert that the domain is not too big.
    //     But to do that, we need the domain min and domain max.
    //     But we can't compute the domain max, because we need the list of particle positions to do that,
    //     and we don't have the list of particle positions because we only store it on the GPU.
    //     We could write a GPU reduction to get the max, but that might be a significant performance hit
    //     just to do an assertion.
    //     On the other hand, we _should_ make this assertion, because otherwise the simulation will start
    //     behaving incorrectly.
    //     Figure out what to do about this.

    {
        // OPTIMIZE doing this interspersion and then un-interspersing them seems like kind of a waste of time?
        //     Although we are accessing the data many times during the merge sort, so maybe worth it.
        //     See if it's faster to sort the permutation array without actually modifying the morton codes
        //     array, by using the permutation data as indices into the morton codes.
        for (u32 i = 0; i < particle_count; i++)
        {
            s->p_scratch_keyval_buffer_1[i].key = s->p_morton_codes[i];
            s->p_scratch_keyval_buffer_1[i].val = i;
        }
        mergeSortMultiThreaded(
            thread_pool,
            s->processor_count,
            particle_count,
            s->p_scratch_keyval_buffer_1,
            s->p_scratch_keyval_buffer_2
        );

        for (u32fast i = 0; i < particle_count; i++)
        {
            // OPTIMIZE Don't bother writing this to p_permutation; we can write it directly to the
            //     host-visible VkBuffer.
            s->p_morton_codes[i] = s->p_scratch_keyval_buffer_1[i].key;
            s->p_permutation[i] = s->p_scratch_keyval_buffer_1[i].val;
        }
    }

    // fill cell list
    {
        ZoneScopedN("fillCellList");

        u32 prev_morton_code = 0;
        if (particle_count > 0)
        {
            s->C_begin[0] = 0;
            prev_morton_code = s->p_morton_codes[0];
        }

        u32fast cell_idx = 1;
        for (u32fast particle_idx = 1; particle_idx < particle_count; particle_idx++)
        {
            u32 morton_code = s->p_morton_codes[particle_idx];
            if (morton_code == prev_morton_code) continue;

            s->C_begin[cell_idx] = (u32)particle_idx;

            prev_morton_code = morton_code;
            cell_idx++;
        }
        s->C_begin[cell_idx] = (u32)particle_count;

        const u32fast cell_count = cell_idx;
        s->cell_count = cell_count;

        for (cell_idx = 0; cell_idx < cell_count; cell_idx++)
        {
            s->C_length[cell_idx] = (u32)s->C_begin[cell_idx+1] - s->C_begin[cell_idx];
        }
    }
    sortCells(
        thread_pool,
        s->processor_count,
        s->cell_count,
        &s->C_begin,
        &s->C_length,
        &s->p_cells_scratch_buffer1,
        &s->p_cells_scratch_buffer2,
        s->p_scratch_keyval_buffer_1,
        s->p_scratch_keyval_buffer_2,
        s->p_morton_codes,
        s->hash_modulus
    );

    {
        ZoneScopedN("fillHashTable");

        for (u32fast i = 0; i < s->hash_modulus; i++) s->H_begin[i] = UINT32_MAX;
        for (u32fast i = 0; i < s->hash_modulus; i++) s->H_length[i] = 0;

        const u32 cell_count = (u32)s->cell_count;

        s->H_begin[0] = 0;
        if (cell_count > 0) s->H_length[0] = 1;
        u32 prev_hash;
        {
            const u32 particle_idx = s->C_begin[0];
            const u32 morton_code = s->p_morton_codes[particle_idx];
            prev_hash = mortonCodeHash(morton_code, s->hash_modulus);
        }
        u32 hash = UINT32_MAX;
        u32 cells_with_this_hash_count = 1;

        for (u32 cell_idx = 1; cell_idx < cell_count; cell_idx++)
        {
            const u32 particle_idx = s->C_begin[cell_idx];
            const u32 morton_code = s->p_morton_codes[particle_idx];
            hash = mortonCodeHash(morton_code, s->hash_modulus);

            if (hash != prev_hash)
            {
                s->H_begin[hash] = cell_idx;
                s->H_length[prev_hash] = cells_with_this_hash_count;

                // OPTIMIZE: if `cells_with_this_hash_count == 1`, set `H_begin[prev_hash] = cell_idx-1`.
                // Make sure you account for this embedding in lookups.

                cells_with_this_hash_count = 0;
                prev_hash = hash;
            }
            cells_with_this_hash_count++;
        }
        if (cell_count > 0) s->H_length[prev_hash] = cells_with_this_hash_count;
    }


    // OPTIMIZE we can probably wait later than here
    if (optional_wait_semaphore != VK_NULL_HANDLE) {

        ZoneScopedN("WaitForUserSemaphore");

        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

        VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &optional_wait_semaphore,
            .pWaitDstStageMask = &wait_stage,
            .commandBufferCount = 0,
            .pCommandBuffers = NULL,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = NULL,
        };
        result = vk_ctx->procs_dev.QueueSubmit(vk_ctx->queue, 1, &submit_info, s->gpu_resources.fence);
        assertVk(result);

        result = vk_ctx->procs_dev.WaitForFences(vk_ctx->device, 1, &s->gpu_resources.fence, true, UINT64_MAX);
        assertVk(result);

        result = vk_ctx->procs_dev.ResetFences(vk_ctx->device, 1, &s->gpu_resources.fence);
        assertVk(result);
    }

    uploadDataToGpu(s, vk_ctx);


    result = vk_ctx->procs_dev.ResetCommandBuffer(s->gpu_resources.general_purpose_command_buffer, 0);
    assertVk(result);

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    result = vk_ctx->procs_dev.BeginCommandBuffer(s->gpu_resources.general_purpose_command_buffer, &begin_info);
    assertVk(result);
    {
        TracyVkZone(vk_ctx->tracy_vk_ctx, s->gpu_resources.general_purpose_command_buffer, "sim::SortAndUpdateParticles");

        // sort particles
        {
            vk_ctx->procs_dev.CmdBindPipeline(
                s->gpu_resources.general_purpose_command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                s->gpu_resources.pipeline_sortParticles
            );

            vk_ctx->procs_dev.CmdBindDescriptorSets(
                s->gpu_resources.general_purpose_command_buffer,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    s->gpu_resources.pipeline_layout_sortParticles,
                    0, // firstSet
                    1, // descriptorSetCount
                    &s->gpu_resources.descriptor_set_main,
                    0, // dynamicOffsetCount
                    NULL // pDynamicOffsets
            );

            vk_ctx->procs_dev.CmdDispatch(
                s->gpu_resources.general_purpose_command_buffer,
                s->gpu_resources.workgroup_count,
                1, // groupCountY
                1 // groupCountZ
            );
        }

        {
            const VkBufferMemoryBarrier buffer_memory_barriers[] {
                {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                    .srcQueueFamilyIndex = vk_ctx->queue_family_index,
                    .dstQueueFamilyIndex = vk_ctx->queue_family_index,
                    .buffer = s->gpu_resources.buffer_positions_sorted.buffer,
                    .offset = 0,
                    .size = VK_WHOLE_SIZE,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                    .srcQueueFamilyIndex = vk_ctx->queue_family_index,
                    .dstQueueFamilyIndex = vk_ctx->queue_family_index,
                    .buffer = s->gpu_resources.buffer_velocities_sorted.buffer,
                    .offset = 0,
                    .size = VK_WHOLE_SIZE,
                },
            };
            constexpr u32 buffer_memory_barrier_count = ARRAY_SIZE(buffer_memory_barriers);

            vk_ctx->procs_dev.CmdPipelineBarrier(
                s->gpu_resources.general_purpose_command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, // dependencyFlags
                0, // memoryBarrierCount
                NULL, // pMemoryBarriers
                buffer_memory_barrier_count,
                buffer_memory_barriers,
                0, // imageMemoryBarrierCount
                NULL // pImageMemoryBarriers
            );
        }

        // update particles
        {
            vk_ctx->procs_dev.CmdBindPipeline(
                s->gpu_resources.general_purpose_command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                s->gpu_resources.pipeline_updateParticles
            );

            const ParticleUpdatePushConstants push_constants {
                .delta_t = delta_t,
                .cell_count = (u32)s->cell_count
            };
            vk_ctx->procs_dev.CmdPushConstants(
                s->gpu_resources.general_purpose_command_buffer,
                s->gpu_resources.pipeline_layout_updateParticles,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0, // offset
                sizeof(ParticleUpdatePushConstants),
                &push_constants
            );

            vk_ctx->procs_dev.CmdBindDescriptorSets(
                s->gpu_resources.general_purpose_command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                s->gpu_resources.pipeline_layout_updateParticles,
                0, // firstSet
                1, // descriptorSetCount
                &s->gpu_resources.descriptor_set_main,
                0, // dynamicOffsetCount
                NULL // pDynamicOffsets
            );

            vk_ctx->procs_dev.CmdDispatch(
                s->gpu_resources.general_purpose_command_buffer,
                s->gpu_resources.workgroup_count, // groupCountX
                1, // groupCountY
                1 // groupCountZ
            );
        }

        TracyVkCollect(vk_ctx->tracy_vk_ctx, s->gpu_resources.general_purpose_command_buffer);
    }
    result = vk_ctx->procs_dev.EndCommandBuffer(s->gpu_resources.general_purpose_command_buffer);
    assertVk(result);

    {
        ZoneScopedN("SubmitParticleUpdateCommandBuffer");

        const u32 signal_semaphore_count =
            (particle_update_finished_signal_semaphore_optional == VK_NULL_HANDLE)
            ? (u32)1
            : (u32)2;

        const VkSemaphore signal_semaphores[2] {
            s->gpu_resources.particle_update_finished_semaphore,
            particle_update_finished_signal_semaphore_optional,
        };

        const VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = NULL,
            .pWaitDstStageMask = 0,
            .commandBufferCount = 1,
            .pCommandBuffers = &s->gpu_resources.general_purpose_command_buffer,
            .signalSemaphoreCount = signal_semaphore_count,
            .pSignalSemaphores = signal_semaphores,
        };
        result = vk_ctx->procs_dev.QueueSubmit(vk_ctx->queue, 1, &submit_info, VK_NULL_HANDLE);
        assertVk(result);
    }

    result = vk_ctx->procs_dev.ResetCommandBuffer(s->gpu_resources.morton_code_command_buffer, 0);
    assertVk(result);

    begin_info = VkCommandBufferBeginInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    result = vk_ctx->procs_dev.BeginCommandBuffer(s->gpu_resources.morton_code_command_buffer, &begin_info);
    assertVk(result);
    {
        TracyVkZone(vk_ctx->tracy_vk_ctx, s->gpu_resources.morton_code_command_buffer, "sim::DomainMinAndMortonCodes");

        PipelineBarrierSrcInfo barrier_src_info =
            recordMinReductionCommands(s, vk_ctx, s->gpu_resources.morton_code_command_buffer);

        VkBufferMemoryBarrier buffer_memory_barrier {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = barrier_src_info.src_access_mask,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = vk_ctx->queue_family_index,
            .dstQueueFamilyIndex = vk_ctx->queue_family_index,
            .buffer = s->gpu_resources.buffer_uniforms.buffer,
            .offset = offsetof(UniformBufferData, domain_min),
            .size = sizeof(vec3),
        };
        vk_ctx->procs_dev.CmdPipelineBarrier(
            s->gpu_resources.morton_code_command_buffer,
            barrier_src_info.src_stage_mask,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, // dependencyFlags
            0, // memoryBarrierCount,
            NULL, // pMemoryBarriers,
            1, // bufferMemoryBarrierCount,
            &buffer_memory_barrier,
            0, // imageMemoryBarrierCount,
            NULL // pImageMemoryBarriers
        );

        vk_ctx->procs_dev.CmdBindPipeline(
            s->gpu_resources.morton_code_command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            s->gpu_resources.pipeline_computeMortonCodes
        );
        vk_ctx->procs_dev.CmdBindDescriptorSets(
            s->gpu_resources.morton_code_command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            s->gpu_resources.pipeline_layout_computeMortonCodes,
            0, // firstSet
            1, // descriptorSetCount
            &s->gpu_resources.descriptor_set_main,
            0, // dynamicOffsetCount
            NULL // pDynamicOffsets
        );
        vk_ctx->procs_dev.CmdDispatch(
            s->gpu_resources.morton_code_command_buffer,
            s->gpu_resources.workgroup_count,
            1, // groupCountY
            1 // groupCountZ
        );

        TracyVkCollect(vk_ctx->tracy_vk_ctx, s->gpu_resources.morton_code_command_buffer);
    }
    result = vk_ctx->procs_dev.EndCommandBuffer(s->gpu_resources.morton_code_command_buffer);
    assertVk(result);

    {
        ZoneScopedN("SubmitMortonCodeCommandBuffer");

        const VkPipelineStageFlags p_wait_dst_stage_mask { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
        const VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &s->gpu_resources.particle_update_finished_semaphore,
            .pWaitDstStageMask = &p_wait_dst_stage_mask,
            .commandBufferCount = 1,
            .pCommandBuffers = &s->gpu_resources.morton_code_command_buffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = NULL,
        };
        result = vk_ctx->procs_dev.QueueSubmit(vk_ctx->queue, 1, &submit_info, s->gpu_resources.fence);
        assertVk(result);
    }
}


extern "C" void getPositionsVertexBuffer(
    const SimData* s,
    VkBuffer* buffer_out,
    VkDeviceSize* buffer_size_out
) {
    *buffer_out = s->gpu_resources.buffer_positions_unsorted.buffer;
    *buffer_size_out = s->particle_count * sizeof(vec4);
}

//
// ===========================================================================================================
//

} // namespace
