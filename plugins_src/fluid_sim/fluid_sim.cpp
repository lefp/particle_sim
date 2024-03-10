#include <cstring>
#include <cstdlib>
#include <cinttypes>
#include <cmath>

#define GLM_FORCE_EXPLICIT_CTOR
#include <glm/glm.hpp>
#include <loguru/loguru.hpp>
#include <tracy/tracy/Tracy.hpp>
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <VulkanMemoryAllocator/vk_mem_alloc.h>

#include "../src/types.hpp"
#include "../src/error_util.hpp"
#include "../src/math_util.hpp"
#include "../src/alloc_util.hpp"
#include "../src/file_util.hpp"
#include "../src/vk_procs.hpp"
#include "../src/vulkan_context.hpp"
#include "../src/defer.hpp"
#include "fluid_sim_types.hpp"

namespace fluid_sim {

using glm::vec3;
using glm::uvec3;

//
// ===========================================================================================================
//

constexpr f32 PI = (f32)M_PI;

constexpr u32 COMPUTE_PIPELINE_DEFAULT_WORKGROUP_SIZE = 128; // TODO FIXME tune?
constexpr struct {
    u32 local_size_x = 0;
} COMPUTE_SHADER_SPECIALIZATION_CONSTANT_IDS;

//
// ===========================================================================================================
//

#define SWAP(a, b) \
{ \
    static_assert(typeof(a) == typeof(b)); \
    \
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


struct PushConstants {
    alignas(16) vec3 domain_min;
    alignas( 4) f32 delta_t;
    alignas( 4) u32 cell_count;
};


static void createDescriptorSet(
    const VulkanContext* vk_ctx,
    const VkBuffer p_buffers[7],
    VkDescriptorSet* descriptor_set_out,
    VkDescriptorSetLayout* layout_out
) {

    VkResult result = VK_ERROR_UNKNOWN;


    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    {
        constexpr u32 binding_count = 7;
        const VkDescriptorSetLayoutBinding bindings[binding_count] {
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            {
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            {
                .binding = 3,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            {
                .binding = 4,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            {
                .binding = 5,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            {
                .binding = 6,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            },
        };

        const VkDescriptorSetLayoutCreateInfo layout_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = binding_count,
            .pBindings = bindings,
        };
        result = vk_ctx->procs_dev.CreateDescriptorSetLayout(vk_ctx->device, &layout_info, NULL, &layout);
        assertVk(result);
    }
    *layout_out = layout;

    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    {
        // TODO FIXME compute this programmatically based on the descriptor binding list, so that you don't
        // need to remember to update this every time you update the bindings.
        constexpr u32 pool_size_count = 2;
        const VkDescriptorPoolSize pool_sizes[pool_size_count] {
            {
                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
            },
            {
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 6,
            },
        };
        const VkDescriptorPoolCreateInfo pool_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = pool_size_count,
            .pPoolSizes = pool_sizes,
        };
        result = vk_ctx->procs_dev.CreateDescriptorPool(vk_ctx->device, &pool_info, NULL, &descriptor_pool);
        assertVk(result);
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    {
        const VkDescriptorSetAllocateInfo alloc_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };
        result = vk_ctx->procs_dev.AllocateDescriptorSets(vk_ctx->device, &alloc_info, &descriptor_set);
        assertVk(result);
    }
    *descriptor_set_out = descriptor_set;

    {
        constexpr u32 write_count = 7;

        const VkDescriptorBufferInfo buffer_infos[write_count] {
            { .buffer = p_buffers[0], .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = p_buffers[1], .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = p_buffers[2], .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = p_buffers[3], .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = p_buffers[4], .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = p_buffers[5], .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = p_buffers[6], .offset = 0, .range = VK_WHOLE_SIZE },
        };

        const VkWriteDescriptorSet writes[write_count] {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &buffer_infos[0],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[1],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[2],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 3,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[3],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 4,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[4],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 5,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[5],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set,
                .dstBinding = 6,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[6],
            },
        };
        vk_ctx->procs_dev.UpdateDescriptorSets(vk_ctx->device, write_count, writes, 0, NULL);
    }
}


static void createComputePipeline(
    const VulkanContext* vk_ctx,
    const u32 workgroup_size,
    const VkDescriptorSetLayout descriptor_set_layout,
    VkPipeline* pipeline_out,
    VkPipelineLayout* pipeline_layout_out
) {

    VkResult result = VK_ERROR_UNKNOWN;


    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    {
        VkPushConstantRange push_constant_range {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants),
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
        void* p_spirv = file_util::readEntireFile("build/shaders/fluid_sim.comp.spv", &spirv_size);
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

    const f32 cell_size = 2.0f * s->parameters.particle_interaction_radius;
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


/// Get the index of the cell that contains the particle.
static inline uvec3 cellIndex(vec3 particle, vec3 domain_min, f32 cell_size_reciprocal) {

    return uvec3((domain_min + particle) * cell_size_reciprocal);
}


static inline u32 cellMortonCode(uvec3 cell_index) {
    return
        ((cell_index.x & 0b0000000001) <<  0) |
        ((cell_index.y & 0b0000000001) <<  1) |
        ((cell_index.z & 0b0000000001) <<  2) |
        ((cell_index.x & 0b0000000010) <<  2) |
        ((cell_index.y & 0b0000000010) <<  3) |
        ((cell_index.z & 0b0000000010) <<  4) |
        ((cell_index.x & 0b0000000100) <<  4) |
        ((cell_index.y & 0b0000000100) <<  5) |
        ((cell_index.z & 0b0000000100) <<  6) |
        ((cell_index.x & 0b0000001000) <<  6) |
        ((cell_index.y & 0b0000001000) <<  7) |
        ((cell_index.z & 0b0000001000) <<  8) |
        ((cell_index.x & 0b0000010000) <<  8) |
        ((cell_index.y & 0b0000010000) <<  9) |
        ((cell_index.z & 0b0000010000) << 10) |
        ((cell_index.x & 0b0000100000) << 10) |
        ((cell_index.y & 0b0000100000) << 11) |
        ((cell_index.z & 0b0000100000) << 12) |
        ((cell_index.x & 0b0001000000) << 12) |
        ((cell_index.y & 0b0001000000) << 13) |
        ((cell_index.z & 0b0001000000) << 14) |
        ((cell_index.x & 0b0010000000) << 14) |
        ((cell_index.y & 0b0010000000) << 15) |
        ((cell_index.z & 0b0010000000) << 16) |
        ((cell_index.x & 0b0100000000) << 16) |
        ((cell_index.y & 0b0100000000) << 17) |
        ((cell_index.z & 0b0100000000) << 18) |
        ((cell_index.x & 0b1000000000) << 18) |
        ((cell_index.y & 0b1000000000) << 19) |
        ((cell_index.z & 0b1000000000) << 20) ;
}


static inline u32 mortonCodeHash(u32 cell_morton_code, u32 hash_modulus) {
    return cell_morton_code % hash_modulus;
}


/// Merge sort the particles by their Morton codes.
/// If the result is written to a scratch buffer, swaps the scratch buffer pointer with the appropriate data
/// buffer pointer.
static void mergeSortByMortonCodes(
    const u64 arr_size,
    vec3 **const pp_positions,
    vec3 **const pp_velocities,
    vec3 **const pp_scratch1,
    vec3 **const pp_scratch2,
    const vec3 domain_min,
    const f32 cell_size_reciprocal
) {

    ZoneScoped;


    vec3* pos_arr1 = *pp_positions;
    vec3* pos_arr2 = *pp_scratch1;

    vec3* vel_arr1 = *pp_velocities;
    vec3* vel_arr2 = *pp_scratch2;

    assert(arr_size >= 2);

    for (u64 bucket_size = 1; bucket_size < arr_size; bucket_size *= 2)
    {
        const u64 bucket_count = arr_size / bucket_size;

        for (u64 bucket_idx = 0; bucket_idx < bucket_count; bucket_idx += 2)
        {
            u64 idx_a = (bucket_idx    ) * bucket_size;
            u64 idx_b = (bucket_idx + 1) * bucket_size;
            u64 idx_dst = idx_a;

            const u64 idx_a_max = glm::min(idx_a + bucket_size, arr_size);
            const u64 idx_b_max = glm::min(idx_b + bucket_size, arr_size);
            const u64 idx_dst_max = glm::min(idx_dst + 2*bucket_size, arr_size);

            for (; idx_dst < idx_dst_max; idx_dst++)
            {
                u32 morton_code_a;
                if (idx_a < idx_a_max)
                {
                    vec3 particle_a = pos_arr1[idx_a];
                    const uvec3 cell_a = cellIndex(particle_a, domain_min, cell_size_reciprocal);
                    morton_code_a = cellMortonCode(cell_a);
                }
                else morton_code_a = UINT32_MAX;

                u32 morton_code_b;
                if (idx_b < idx_b_max)
                {
                    vec3 particle_b = pos_arr1[idx_b];
                    const uvec3 cell_b = cellIndex(particle_b, domain_min, cell_size_reciprocal);
                    morton_code_b = cellMortonCode(cell_b);
                }
                else morton_code_b = UINT32_MAX;

                if (morton_code_a <= morton_code_b)
                {
                    pos_arr2[idx_dst] = pos_arr1[idx_a];
                    vel_arr2[idx_dst] = vel_arr1[idx_a];
                    idx_a++;
                }
                else
                {
                    pos_arr2[idx_dst] = pos_arr1[idx_b];
                    vel_arr2[idx_dst] = vel_arr1[idx_b];
                    idx_b++;
                }
            }
        }

        vec3* tmp = pos_arr1;
        pos_arr1 = pos_arr2;
        pos_arr2 = tmp;

        tmp = vel_arr1;
        vel_arr1 = vel_arr2;
        vel_arr2 = tmp;
    }

    *pp_positions = pos_arr1;
    *pp_velocities = vel_arr1;
    *pp_scratch1 = pos_arr2;
    *pp_scratch2 = vel_arr2;
}


static void mergeSortByCellHashes(
    const u64 array_size,
    u32 **const pp_cells,
    u32 **const pp_lengths,
    u32 **const pp_scratch1,
    u32 **const pp_scratch2,
    const vec3 *const p_particles,
    const vec3 domain_min,
    const f32 cell_size_reciprocal,
    const u32 hash_modulus
) {

    ZoneScoped;


    if (array_size < 2) return;

    u32* cells_arr1 = *pp_cells;
    u32* cells_arr2 = *pp_scratch1;

    u32* lens_arr1 = *pp_lengths;
    u32* lens_arr2 = *pp_scratch2;

    for (u64 bucket_size = 1; bucket_size < array_size; bucket_size *= 2)
    {
        const u64 bucket_count = array_size / bucket_size;

        for (u64 bucket_idx = 0; bucket_idx < bucket_count; bucket_idx += 2)
        {
            u64 idx_a = (bucket_idx    ) * bucket_size;
            u64 idx_b = (bucket_idx + 1) * bucket_size;
            u64 idx_dst = idx_a;

            const u64 idx_a_max = glm::min(idx_a + bucket_size, array_size);
            const u64 idx_b_max = glm::min(idx_b + bucket_size, array_size);
            const u64 idx_dst_max = glm::min(idx_dst + 2*bucket_size, array_size);

            for (; idx_dst < idx_dst_max; idx_dst++)
            {
                u32 cell_hash_a;
                if (idx_a < idx_a_max) {
                    const u32 particle_idx = cells_arr1[idx_a];
                    const uvec3 cell = cellIndex(p_particles[particle_idx], domain_min, cell_size_reciprocal);
                    const u32 morton_code = cellMortonCode(cell);
                    cell_hash_a = mortonCodeHash(morton_code, hash_modulus);
                }
                else cell_hash_a = UINT32_MAX;

                u32 cell_hash_b;
                if (idx_b < idx_b_max) {
                    const u32 particle_idx = cells_arr1[idx_b];
                    const uvec3 cell = cellIndex(p_particles[particle_idx], domain_min, cell_size_reciprocal);
                    const u32 morton_code = cellMortonCode(cell);
                    cell_hash_b = mortonCodeHash(morton_code, hash_modulus);
                }
                else cell_hash_b = UINT32_MAX;

                if (cell_hash_a <= cell_hash_b) {
                    cells_arr2[idx_dst] = cells_arr1[idx_a];
                    lens_arr2[idx_dst] = lens_arr1[idx_a];
                    idx_a++;
                }
                else {
                    cells_arr2[idx_dst] = cells_arr1[idx_b];
                    lens_arr2[idx_dst] = lens_arr1[idx_b];
                    idx_b++;
                }
            }
        }

        u32* tmp = cells_arr1;
        cells_arr1 = cells_arr2;
        cells_arr2 = tmp;

        tmp = lens_arr1;
        lens_arr1 = lens_arr2;
        lens_arr2 = tmp;
    }

    *pp_cells = cells_arr1;
    *pp_lengths = lens_arr1;
    *pp_scratch1 = cells_arr2;
    *pp_scratch2 = lens_arr2;
}


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


struct CompactCell {
    u32 first_particle_idx;
    u32 particle_count;
};


static inline CompactCell cell3dToCell(const SimData* s, const uvec3 cell_idx_3d, const vec3 domain_min) {

    const u32 morton_code = cellMortonCode(cell_idx_3d);
    const u32 hash = mortonCodeHash(morton_code, s->hash_modulus);

    const u32 first_cell_with_hash_idx = s->H_begin[hash];
    const u32 n_cells_with_hash = s->H_length[hash];

    if (n_cells_with_hash == 0) return CompactCell { .first_particle_idx = UINT32_MAX, .particle_count = 0 };

    u32 cell_idx = first_cell_with_hash_idx;
    const u32 cell_idx_end = cell_idx + n_cells_with_hash;

    for (; cell_idx < cell_idx_end; cell_idx++)
    {
        const u32 first_particle_in_cell_idx = s->p_cells[cell_idx];
        assert(first_particle_in_cell_idx < s->particle_count);

        const vec3 first_particle_in_cell = s->p_positions[first_particle_in_cell_idx];
        if (
            cellMortonCode(cellIndex(first_particle_in_cell, domain_min, s->parameters.cell_size_reciprocal))
            == morton_code
        ) {
            return CompactCell {
                .first_particle_idx = first_particle_in_cell_idx,
                .particle_count = s->p_cell_lengths[cell_idx],
            };
        }
    }

    return CompactCell { .first_particle_idx = UINT32_MAX, .particle_count = 0 };
}


static inline CompactCell particleToCell(const SimData* s, const vec3 particle, const vec3 domain_min) {

    const uvec3 cell_idx_3d = cellIndex(particle, domain_min, s->parameters.cell_size_reciprocal);
    return cell3dToCell(s, cell_idx_3d, domain_min);
}


static inline vec3 accelerationDueToParticlesInCell(
    const SimData* s,
    const u32fast target_particle_idx,
    const uvec3 cell_idx_3d,
    const vec3 domain_min
) {

    const CompactCell cell = cell3dToCell(s, cell_idx_3d, domain_min);
    if (cell.particle_count == 0) return vec3(0.0f); // cell doesn't exist

    const vec3 pos = s->p_positions[target_particle_idx];

    vec3 accel {};

    u32fast i = cell.first_particle_idx;
    const u32fast i_end = i + cell.particle_count;

    for (; i < i_end; i++)
    {
        // OPTIMIZE: we can remove this check if we know that none of the particles are the target particle.
        //     E.g. if the particle list comes from a different cell than the target particle.
        if (i == target_particle_idx) continue;

        vec3 disp = s->p_positions[i] - pos;
        f32 dist = glm::length(disp);

        if (dist >= s->parameters.particle_interaction_radius) continue;
        if (dist < 1e-7)
        {
            LOG_F(WARNING, "distance too small: %" PRIuFAST32 " %" PRIuFAST32 " %f", target_particle_idx, i, dist);
            continue;
        }
        vec3 disp_unit = disp / dist;

        accel += s->parameters.spring_stiffness * (dist - s->parameters.spring_rest_length) * disp_unit;
    }

    return accel;
}


struct UniformBufferData {

    alignas(4) f32 rest_particle_density;
    alignas(4) f32 particle_interaction_radius;
    alignas(4) f32 spring_rest_length;
    alignas(4) f32 spring_stiffness;
    alignas(4) f32 cell_size_reciprocal;

    // stuff whose lifetime is the lifetime of the sim
    alignas(4) u32 particle_count;
    alignas(4) u32 hash_modulus;
};

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
        VkCommandBufferAllocateInfo alloc_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = resources.command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        result = vk_ctx->procs_dev.AllocateCommandBuffers(vk_ctx->device, &alloc_info, &resources.command_buffer);
        assertVk(result);
    }

    {
        const VkFenceCreateInfo fence_info { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        result = vk_ctx->procs_dev.CreateFence(vk_ctx->device, &fence_info, NULL, &resources.fence);
        assertVk(result);
    }


    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = sizeof(UniformBufferData),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_uniforms, &resources.allocation_uniforms, &resources.allocation_info_uniforms
        );
        assertVk(result);
    }

    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = particle_count * sizeof(vec3),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // OPTIMIZE use a staging buffer instead?
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_positions, &resources.allocation_positions, &resources.allocation_info_positions
        );
        assertVk(result);
    }

    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = particle_count * sizeof(vec3),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // OPTIMIZE use a staging buffer instead?
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_velocities, &resources.allocation_velocities, &resources.allocation_info_velocities
        );
        assertVk(result);
    }

    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = (particle_count + 1) * sizeof(u32),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // OPTIMIZE use a staging buffer instead?
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_C_begin, &resources.allocation_C_begin, &resources.allocation_info_C_begin
        );
        assertVk(result);
    }

    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = particle_count * sizeof(u32),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // OPTIMIZE use a staging buffer instead?
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_C_length, &resources.allocation_C_length, &resources.allocation_info_C_length
        );
        assertVk(result);
    }

    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = hash_modulus * sizeof(u32),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // OPTIMIZE use a staging buffer instead?
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_H_begin, &resources.allocation_H_begin, &resources.allocation_info_H_begin
        );
        assertVk(result);
    }

    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = hash_modulus * sizeof(u32),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // OPTIMIZE use a staging buffer instead?
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_H_length, &resources.allocation_H_length, &resources.allocation_info_H_length
        );
        assertVk(result);
    }


    {
        const VkBuffer buffers[7] {
            resources.buffer_uniforms,
            resources.buffer_positions,
            resources.buffer_velocities,
            resources.buffer_C_begin,
            resources.buffer_C_length,
            resources.buffer_H_begin,
            resources.buffer_H_length,
        };
        createDescriptorSet(vk_ctx, buffers, &resources.descriptor_set, &resources.descriptor_set_layout);
    }

    createComputePipeline(
        vk_ctx,
        workgroup_size,
        resources.descriptor_set_layout,
        &resources.pipeline,
        &resources.pipeline_layout
    );


    return resources;
}


static void uploadBufferToHostVisibleGpuMemory(
    const VulkanContext* vk_ctx,
    const u32fast size_bytes,
    const void* src,
    const VmaAllocation dst
) {

    VkResult result = VK_ERROR_UNKNOWN;


    void* p_mapped_memory = NULL;
    {
        result = vmaMapMemory(vk_ctx->vma_allocator, dst, &p_mapped_memory);
        assertVk(result);
    }

    memcpy(p_mapped_memory, src, size_bytes);

    result = vmaFlushAllocation(vk_ctx->vma_allocator, dst, 0, size_bytes);
    assertVk(result);

    vmaUnmapMemory(vk_ctx->vma_allocator, dst);
}


static void downloadBufferFromHostVisibleGpuMemory(
    const VulkanContext* vk_ctx,
    const u32fast size_bytes,
    const VmaAllocation src,
    void* dst
) {

    VkResult result = VK_ERROR_UNKNOWN;


    const void* p_mapped_memory = NULL;
    {
        void* ptr = NULL;

        result = vmaMapMemory(vk_ctx->vma_allocator, src, &ptr);
        assertVk(result);

        p_mapped_memory = ptr;
    }

    memcpy(dst, p_mapped_memory, size_bytes);

    result = vmaFlushAllocation(vk_ctx->vma_allocator, src, 0, size_bytes);
    assertVk(result);

    vmaUnmapMemory(vk_ctx->vma_allocator, src);
}


static void uploadDataToGpu(const SimData* s, const VulkanContext* vk_ctx) {

    {
        const UniformBufferData uniform_data {
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
            s->gpu_resources.allocation_uniforms
        );
    }

    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->particle_count * sizeof(*s->p_positions),
        s->p_positions,
        s->gpu_resources.allocation_positions
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->particle_count * sizeof(*s->p_velocities),
        s->p_velocities,
        s->gpu_resources.allocation_velocities
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->cell_count * sizeof(*s->p_cells),
        s->p_cells,
        s->gpu_resources.allocation_C_begin
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->cell_count * sizeof(*s->p_cell_lengths),
        s->p_cell_lengths,
        s->gpu_resources.allocation_C_length
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->hash_modulus * sizeof(*s->H_begin),
        s->H_begin,
        s->gpu_resources.allocation_H_begin
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->hash_modulus * sizeof(*s->H_length),
        s->H_length,
        s->gpu_resources.allocation_H_length
    );
}


static void downloadDataFromGpu(SimData* s, const VulkanContext* vk_ctx) {

    downloadBufferFromHostVisibleGpuMemory(
        vk_ctx,
        s->particle_count * sizeof(*s->p_positions),
        s->gpu_resources.allocation_positions,
        s->p_positions
    );
    downloadBufferFromHostVisibleGpuMemory(
        vk_ctx,
        s->particle_count * sizeof(*s->p_velocities),
        s->gpu_resources.allocation_velocities,
        s->p_velocities
    );
}


extern "C" SimData create(
    const SimParameters* params,
    const VulkanContext* vk_ctx,
    u32fast particle_count,
    const vec3* p_initial_positions
) {

    ZoneScoped;

    SimData s {};
    {
        s.particle_count = particle_count;

        s.p_positions = mallocArray(particle_count, vec3);
        memcpy(s.p_positions, p_initial_positions, particle_count * sizeof(vec3));

        s.p_velocities = callocArray(particle_count, vec3);

        s.p_particles_scratch_buffer1 = callocArray(particle_count, vec3);
        s.p_particles_scratch_buffer2 = callocArray(particle_count, vec3);

        s.cell_count = 0;
        s.p_cells = callocArray(particle_count + 1, u32);
        s.p_cell_lengths = callocArray(particle_count, u32);

        s.p_cells_scratch_buffer1 = callocArray(particle_count + 1, u32);
        s.p_cells_scratch_buffer2 = callocArray(particle_count + 1, u32);

        // smallest prime number larger than the maximum number of particles
        // OPTIMIZE profile this and optimize if too slow
        u32fast hash_modulus = getNextPrimeNumberExclusive(particle_count);
        assert(hash_modulus <= UINT32_MAX);
        s.hash_modulus = (u32)hash_modulus;

        s.H_begin = callocArray(hash_modulus, u32);
        s.H_length = callocArray(hash_modulus, u32);

        setParams(&s, params);

        s.gpu_resources = createGpuResources(vk_ctx, particle_count, hash_modulus);
    }

    LOG_F(INFO, "Initialized fluid sim with %" PRIuFAST32 " particles.", s.particle_count);

    return s;
}


extern "C" void destroy(SimData* s) {

    // TODO FIXME free all the other arrays you allocated
    // TODO FIXME destroy all the GPU resources

    free(s->p_positions);
    free(s->p_velocities);
    s->particle_count = 0;
}


extern "C" void advance(SimData* s, const VulkanContext* vk_ctx, f32 delta_t) {

    ZoneScoped;


    assert(delta_t > 1e-5); // assert nonzero

    const u32fast particle_count = s->particle_count;
    const f32 cell_size_reciprocal = s->parameters.cell_size_reciprocal;

    vec3 domain_min = vec3(INFINITY);
    vec3 domain_max = vec3(-INFINITY);
    for (u32fast i = 0; i < particle_count; i++)
    {
        domain_min = glm::min(s->p_positions[i], domain_min);
        domain_max = glm::max(s->p_positions[i], domain_max);
    }

    {
        uvec3 cell_count = uvec3(glm::ceil((domain_max - domain_min) * cell_size_reciprocal) + 0.5f);
        // 32-bit Morton codes can handle at most a 1024x1024x1024 grid.
        // If this turns out to be insufficient, consider using 64-bit Morton codes.
        assert(cell_count.x < 1024 and cell_count.y < 1024 and cell_count.z < 1024);
        (void)cell_count; // to prevent "unused variable" complaints when compiling with NDEBUG
    }

    mergeSortByMortonCodes(
        particle_count,
        &s->p_positions, &s->p_velocities, &s->p_particles_scratch_buffer1, &s->p_particles_scratch_buffer2,
        domain_min, cell_size_reciprocal
    );

    // fill cell list
    {
        ZoneScopedN("fillCellList");

        u32 prev_morton_code = 0;
        if (particle_count > 0)
        {
            s->p_cells[0] = 0;
            prev_morton_code = cellMortonCode(cellIndex(s->p_positions[0], domain_min, cell_size_reciprocal));
        }

        u32fast cell_idx = 1;
        for (u32fast particle_idx = 1; particle_idx < particle_count; particle_idx++)
        {
            u32 morton_code =
                cellMortonCode(cellIndex(s->p_positions[particle_idx], domain_min, cell_size_reciprocal));

            if (morton_code == prev_morton_code) continue;

            prev_morton_code = morton_code;

            s->p_cells[cell_idx] = (u32)particle_idx;

            cell_idx++;
        }
        s->p_cells[cell_idx] = (u32)particle_count;

        const u32fast cell_count = cell_idx;
        s->cell_count = cell_count;

        for (cell_idx = 0; cell_idx < cell_count; cell_idx++)
        {
            s->p_cell_lengths[cell_idx] = (u32)s->p_cells[cell_idx+1] - s->p_cells[cell_idx];
        }

        mergeSortByCellHashes(
            cell_count,
            &s->p_cells, &s->p_cell_lengths, &s->p_cells_scratch_buffer1, &s->p_cells_scratch_buffer2,
            s->p_positions,
            domain_min, cell_size_reciprocal,
            s->hash_modulus
        );
    }

    {
        ZoneScopedN("fillHashTable");

        for (u32fast i = 0; i < s->hash_modulus; i++) s->H_begin[i] = UINT32_MAX;
        for (u32fast i = 0; i < s->hash_modulus; i++) s->H_length[i] = 0;

        const u32 cell_count = (u32)s->cell_count;

        s->H_begin[0] = 0;
        if (cell_count > 0) s->H_length[0] = 1;
        u32 prev_hash;
        {
            const u32 particle_idx = s->p_cells[0];
            const vec3 particle = s->p_positions[particle_idx];
            const uvec3 cell_idx_3d = cellIndex(particle, domain_min, cell_size_reciprocal);
            const u32 morton_code = cellMortonCode(cell_idx_3d);
            prev_hash = mortonCodeHash(morton_code, s->hash_modulus);
        }
        u32 hash = UINT32_MAX;
        u32 cells_with_this_hash_count = 1;

        for (u32 cell_idx = 1; cell_idx < cell_count; cell_idx++)
        {
            const u32 particle_idx = s->p_cells[cell_idx];
            const vec3 particle = s->p_positions[particle_idx];
            const uvec3 cell_idx_3d = cellIndex(particle, domain_min, cell_size_reciprocal);
            const u32 morton_code = cellMortonCode(cell_idx_3d);
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


    uploadDataToGpu(s, vk_ctx);
    // TODO: if a renderer directly uses our positions buffer, wait for the renderer to finish

    {
        VkResult result = vk_ctx->procs_dev.ResetCommandBuffer(s->gpu_resources.command_buffer, 0);
        assertVk(result);

        VkCommandBufferBeginInfo begin_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        result = vk_ctx->procs_dev.BeginCommandBuffer(s->gpu_resources.command_buffer, &begin_info);
        assertVk(result);

        const PushConstants push_constants {
            .domain_min = domain_min,
            .delta_t = delta_t,
            .cell_count = (u32)s->cell_count
        };
        vk_ctx->procs_dev.CmdPushConstants(
            s->gpu_resources.command_buffer,
            s->gpu_resources.pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0, // offset
            sizeof(PushConstants),
            &push_constants
        );

        vk_ctx->procs_dev.CmdBindDescriptorSets(
            s->gpu_resources.command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            s->gpu_resources.pipeline_layout,
            0, // firstSet
            1, // descriptorSetCount
            &s->gpu_resources.descriptor_set,
            0, // dynamicOffsetCount
            NULL // pDynamicOffsets
        );

        vk_ctx->procs_dev.CmdBindPipeline(
            s->gpu_resources.command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            s->gpu_resources.pipeline
        );

        vk_ctx->procs_dev.CmdDispatch(
            s->gpu_resources.command_buffer,
            s->gpu_resources.workgroup_count, // groupCountX
            1, // groupCountY
            1 // groupCountZ
        );

        result = vk_ctx->procs_dev.EndCommandBuffer(s->gpu_resources.command_buffer);
        assertVk(result);

        const VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = NULL,
            .pWaitDstStageMask = 0,
            .commandBufferCount = 1,
            .pCommandBuffers = &s->gpu_resources.command_buffer,
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

    downloadDataFromGpu(s, vk_ctx);
};

//
// ===========================================================================================================
//

} // namespace
