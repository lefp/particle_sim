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


struct PushConstants {
    alignas(16) vec3 domain_min;
    alignas( 4) f32 delta_t;
    alignas( 4) u32 cell_count;
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


static void createDescriptorStuff(
    GpuResources* res,
    const VulkanContext* vk_ctx
) {
    using descriptor_management::DescriptorSetLayout;

    constexpr u32 layout_binding_count_main = 7;
    VkDescriptorSetLayoutBinding layout_bindings_main[layout_binding_count_main];
    {
        layout_bindings_main[0] = VkDescriptorSetLayoutBinding {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
        for (u32 i = 1; i < layout_binding_count_main; i++)
        {
            layout_bindings_main[i] = VkDescriptorSetLayoutBinding {
                .binding = i,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            };
        }
    }

    constexpr u32 layout_binding_count_reduction = 2;
    VkDescriptorSetLayoutBinding layout_bindings_reduction[layout_binding_count_reduction] {
        VkDescriptorSetLayoutBinding {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        VkDescriptorSetLayoutBinding {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };

    constexpr u32fast layout_count = 2;
    const DescriptorSetLayout layout_infos[layout_count] {
        DescriptorSetLayout {
            .binding_count = layout_binding_count_main,
            .p_bindings = layout_bindings_main,
        },
        DescriptorSetLayout {
            .binding_count = layout_binding_count_reduction,
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
        const VkDescriptorBufferInfo buffer_infos[layout_binding_count_main] {
            { .buffer = res->buffer_uniforms.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = res->buffer_positions.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = res->buffer_velocities.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = res->buffer_C_begin.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = res->buffer_C_length.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = res->buffer_H_begin.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
            { .buffer = res->buffer_H_length.buffer, .offset = 0, .range = VK_WHOLE_SIZE },
        };

        VkWriteDescriptorSet writes[layout_binding_count_main] {};
        {
            for (u32 binding_idx = 0; binding_idx < layout_binding_count_main; binding_idx++)
            {
                writes[binding_idx] = VkWriteDescriptorSet {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = res->descriptor_set_main,
                    .dstBinding = binding_idx,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = layout_bindings_main[binding_idx].descriptorType,
                    .pBufferInfo = &buffer_infos[binding_idx],
                };
            }
        }

        vk_ctx->procs_dev.UpdateDescriptorSets(vk_ctx->device, layout_binding_count_main, writes, 0, NULL);
    }

    {
        VkDescriptorBufferInfo buffer_info_positions
            { .buffer = res->buffer_positions.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
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


static void createComputePipeline(
    const VulkanContext* vk_ctx,
    const char *const spirv_filepath,
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
        void* p_spirv = file_util::readEntireFile(spirv_filepath, &spirv_size);
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


static void sortParticles(

    thread_pool::ThreadPool* thread_pool,
    u32 thread_count,

    const vec3 domain_min,
    const f32 cell_size_reciprocal,

    const u32fast particle_count,

    vec4** pp_positions,
    vec4** pp_velocities,

    vec4** pp_positions_scratch,
    vec4** pp_velocities_scratch,

    KeyVal *const p_scratch1,
    KeyVal *const p_scratch2
) {

    ZoneScoped;

    KeyVal* p_keyvals = p_scratch1;
    KeyVal* p_scratch = p_scratch2;

    vec4* p_positions_in = *pp_positions;
    vec4* p_velocities_in = *pp_velocities;
    vec4* p_positions_out = *pp_positions_scratch;
    vec4* p_velocities_out = *pp_velocities_scratch;

    {
        ZoneScopedN("init morton codes");

        for (u32 i = 0; i < particle_count; i++)
        {
            p_keyvals[i].key =
                cellMortonCode(cellIndex(vec3(p_positions_in[i]), domain_min, cell_size_reciprocal));
            p_keyvals[i].val = i;
        }
    }

    alwaysAssert(thread_count > 0);
    mergeSortMultiThreaded(
        thread_pool,
        thread_count,
        particle_count,
        p_keyvals,
        p_scratch
    );

    {
        ZoneScopedN("sort by permutation");

        for (u32fast i = 0; i < particle_count; i++)
        {
            u32fast src_idx = p_keyvals[i].val;

            p_positions_out[i] = p_positions_in[src_idx];
            p_positions_out[i] = p_positions_in[src_idx];

            p_velocities_out[i] = p_velocities_in[src_idx];
            p_velocities_out[i] = p_velocities_in[src_idx];
        }
    }

    *pp_positions = p_positions_out;
    *pp_velocities = p_velocities_out;
    *pp_positions_scratch = p_positions_in;
    *pp_velocities_scratch = p_velocities_in;
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

    // TODO OPTIMIZE: this might be faster if we take `p_cell_morton_codes` instead of `p_particles`.
    const vec4 *const p_particles,

    const vec3 domain_min,
    const f32 cell_size_reciprocal,
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
            const uvec3 cell = cellIndex(vec3(p_particles[particle_idx]), domain_min, cell_size_reciprocal);
            const u32 morton_code = cellMortonCode(cell);
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
        const u32 first_particle_in_cell_idx = s->C_begin[cell_idx];
        assert(first_particle_in_cell_idx < s->particle_count);

        const vec3 first_particle_in_cell = vec3(s->p_positions[first_particle_in_cell_idx]);
        if (
            cellMortonCode(cellIndex(first_particle_in_cell, domain_min, s->parameters.cell_size_reciprocal))
            == morton_code
        ) {
            return CompactCell {
                .first_particle_idx = first_particle_in_cell_idx,
                .particle_count = s->C_length[cell_idx],
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

    const vec3 pos = vec3(s->p_positions[target_particle_idx]);

    vec3 accel {};

    u32fast i = cell.first_particle_idx;
    const u32fast i_end = i + cell.particle_count;

    for (; i < i_end; i++)
    {
        // OPTIMIZE: we can remove this check if we know that none of the particles are the target particle.
        //     E.g. if the particle list comes from a different cell than the target particle.
        if (i == target_particle_idx) continue;

        vec3 disp = vec3(s->p_positions[i]) - pos;
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
            &resources.buffer_uniforms.buffer,
            &resources.buffer_uniforms.allocation,
            &resources.buffer_uniforms.allocation_info
        );
        assertVk(result);
    }

    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = particle_count * sizeof(vec4), // vec4 because std430 alignment
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_positions.buffer,
            &resources.buffer_positions.allocation,
            &resources.buffer_positions.allocation_info
        );
        assertVk(result);
    }
    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = particle_count * sizeof(vec4), // vec4 because std430 alignment
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, // hints that caching should be enabled
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_staging_positions.buffer,
            &resources.buffer_staging_positions.allocation,
            &resources.buffer_staging_positions.allocation_info
        );
        assertVk(result);
    }

    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = particle_count * sizeof(vec4), // vec4 because std430 alignment
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_velocities.buffer,
            &resources.buffer_velocities.allocation,
            &resources.buffer_velocities.allocation_info
        );
        assertVk(result);
    }
    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = particle_count * sizeof(vec4), // vec4 because std430 alignment
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, // hints that caching should be enabled
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_staging_velocities.buffer,
            &resources.buffer_staging_velocities.allocation,
            &resources.buffer_staging_velocities.allocation_info
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
            &resources.buffer_C_begin.buffer,
            &resources.buffer_C_begin.allocation,
            &resources.buffer_C_begin.allocation_info
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
            &resources.buffer_C_length.buffer,
            &resources.buffer_C_length.allocation,
            &resources.buffer_C_length.allocation_info
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
            &resources.buffer_H_begin.buffer,
            &resources.buffer_H_begin.allocation,
            &resources.buffer_H_begin.allocation_info
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
            &resources.buffer_H_length.buffer,
            &resources.buffer_H_length.allocation,
            &resources.buffer_H_length.allocation_info
        );
        assertVk(result);
    }

    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = particle_count * sizeof(vec4), // OPTIMIZE this can be smaller (divCeil(size, 2)?)
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_reduction_1.buffer,
            &resources.buffer_reduction_1.allocation,
            &resources.buffer_reduction_1.allocation_info
        );
        assertVk(result);
    }

    {
        VkBufferCreateInfo buffer_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = particle_count * sizeof(vec4), // OPTIMIZE this can be smaller (divCeil(size, 4)?)
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &vk_ctx->queue_family_index,
        };
        VmaAllocationCreateInfo buffer_alloc_info {
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        };
        result = vmaCreateBuffer(
            vk_ctx->vma_allocator, &buffer_info, &buffer_alloc_info,
            &resources.buffer_reduction_2.buffer,
            &resources.buffer_reduction_2.allocation,
            &resources.buffer_reduction_2.allocation_info
        );
        assertVk(result);
    }

    createDescriptorStuff(&resources, vk_ctx);

    createComputePipeline(
        vk_ctx,
        "build/shaders/fluidSim_updateVelocities.comp.spv",
        workgroup_size,
        resources.descriptor_set_layout_main,
        &resources.pipeline_updatePositions,
        &resources.pipeline_layout_updatePositions
    );
    createComputePipeline(
        vk_ctx,
        "build/shaders/fluidSim_updatePositions.comp.spv",
        workgroup_size,
        resources.descriptor_set_layout_main,
        &resources.pipeline_updateVelocities,
        &resources.pipeline_layout_updateVelocities
    );


    return resources;
}


static void destroyGpuResources(GpuResources* res, const VulkanContext* vk_ctx) {

    ZoneScoped;

    VkResult result = vk_ctx->procs_dev.QueueWaitIdle(vk_ctx->queue);
    assertVk(result);

    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_uniforms.buffer, res->buffer_uniforms.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_positions.buffer, res->buffer_positions.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_staging_positions.buffer, res->buffer_staging_positions.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_velocities.buffer, res->buffer_velocities.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_staging_velocities.buffer, res->buffer_staging_velocities.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_C_begin.buffer, res->buffer_C_begin.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_C_length.buffer, res->buffer_C_length.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_H_begin.buffer, res->buffer_H_begin.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_H_length.buffer, res->buffer_H_length.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_reduction_1.buffer, res->buffer_reduction_1.allocation);
    vmaDestroyBuffer(vk_ctx->vma_allocator, res->buffer_reduction_2.buffer, res->buffer_reduction_2.allocation);

    vk_ctx->procs_dev.FreeCommandBuffers(vk_ctx->device, res->command_pool, 1, &res->command_buffer);
    vk_ctx->procs_dev.DestroyCommandPool(vk_ctx->device, res->command_pool, NULL);

    vk_ctx->procs_dev.DestroyDescriptorSetLayout(vk_ctx->device, res->descriptor_set_layout_main, NULL);
    vk_ctx->procs_dev.DestroyDescriptorSetLayout(vk_ctx->device, res->descriptor_set_layout_reduction, NULL);
    vk_ctx->procs_dev.DestroyDescriptorPool(vk_ctx->device, res->descriptor_pool, NULL);

    vk_ctx->procs_dev.DestroyPipeline(vk_ctx->device, res->pipeline_updatePositions, NULL);
    vk_ctx->procs_dev.DestroyPipelineLayout(vk_ctx->device, res->pipeline_layout_updatePositions, NULL);

    vk_ctx->procs_dev.DestroyPipeline(vk_ctx->device, res->pipeline_updateVelocities, NULL);
    vk_ctx->procs_dev.DestroyPipelineLayout(vk_ctx->device, res->pipeline_layout_updateVelocities, NULL);

    vk_ctx->procs_dev.DestroyFence(vk_ctx->device, res->fence, NULL);
}


static void uploadBufferToHostVisibleGpuMemory(
    const VulkanContext* vk_ctx,
    const u32fast size_bytes,
    const void* src,
    const VmaAllocation dst
) {

    ZoneScoped;

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

    result = vmaInvalidateAllocation(vk_ctx->vma_allocator, src, 0, size_bytes);
    assertVk(result);

    {

        ZoneScopedN("memcpy");
        memcpy(dst, p_mapped_memory, size_bytes);
    }

    // TODO FIXME: Why flush? You didn't modify the data.
    result = vmaFlushAllocation(vk_ctx->vma_allocator, src, 0, size_bytes);
    assertVk(result);

    vmaUnmapMemory(vk_ctx->vma_allocator, src);
}


static void uploadDataToGpu(const SimData* s, const VulkanContext* vk_ctx) {

    ZoneScoped;

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
            s->gpu_resources.buffer_uniforms.allocation
        );
    }

    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->particle_count * sizeof(vec4),
        s->p_positions,
        s->gpu_resources.buffer_staging_positions.allocation
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->particle_count * sizeof(vec4),
        s->p_velocities,
        s->gpu_resources.buffer_staging_velocities.allocation
    );

    {
        VkCommandBufferBeginInfo begin_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VkResult result = vk_ctx->procs_dev.BeginCommandBuffer(s->gpu_resources.command_buffer, &begin_info);
        assertVk(result);

        VkBufferCopy copy_info_positions {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = s->gpu_resources.buffer_staging_positions.allocation_info.size,
        };
        vk_ctx->procs_dev.CmdCopyBuffer(
            s->gpu_resources.command_buffer,
            s->gpu_resources.buffer_staging_positions.buffer,
            s->gpu_resources.buffer_positions.buffer,
            1, // regionCount
            &copy_info_positions
        );

        VkBufferCopy copy_info_velocities {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = s->gpu_resources.buffer_staging_velocities.allocation_info.size,
        };
        vk_ctx->procs_dev.CmdCopyBuffer(
            s->gpu_resources.command_buffer,
            s->gpu_resources.buffer_staging_velocities.buffer,
            s->gpu_resources.buffer_velocities.buffer,
            1, // regionCount
            &copy_info_velocities
        );

        result = vk_ctx->procs_dev.EndCommandBuffer(s->gpu_resources.command_buffer);
        assertVk(result);

        VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = NULL,
            .pWaitDstStageMask = NULL,
            .commandBufferCount = 1,
            .pCommandBuffers = &s->gpu_resources.command_buffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = NULL,
        };
        result = vk_ctx->procs_dev.QueueSubmit(vk_ctx->queue, 1, &submit_info, s->gpu_resources.fence);
        assertVk(result);
    }

    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->cell_count * sizeof(*s->C_begin),
        s->C_begin,
        s->gpu_resources.buffer_C_begin.allocation
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->cell_count * sizeof(*s->C_length),
        s->C_length,
        s->gpu_resources.buffer_C_length.allocation
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->hash_modulus * sizeof(*s->H_begin),
        s->H_begin,
        s->gpu_resources.buffer_H_begin.allocation
    );
    uploadBufferToHostVisibleGpuMemory(
        vk_ctx,
        s->hash_modulus * sizeof(*s->H_length),
        s->H_length,
        s->gpu_resources.buffer_H_length.allocation
    );

    // OPTIMIZE:
    //     Instead of signaling a fence in the queue submission and waiting for it here, we can signal a
    //     semaphore that the compute shader dispatch waits for.
    VkResult result =
        vk_ctx->procs_dev.WaitForFences(vk_ctx->device, 1, &s->gpu_resources.fence, VK_TRUE, UINT64_MAX);
    assertVk(result);

    result = vk_ctx->procs_dev.ResetFences(vk_ctx->device, 1, &s->gpu_resources.fence);
    assertVk(result);
}


static void downloadDataFromGpu(SimData* s, const VulkanContext* vk_ctx) {

    ZoneScoped;

    {
        VkCommandBufferBeginInfo begin_info {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VkResult result = vk_ctx->procs_dev.BeginCommandBuffer(s->gpu_resources.command_buffer, &begin_info);
        assertVk(result);

        VkBufferCopy copy_info_positions {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = s->gpu_resources.buffer_staging_positions.allocation_info.size,
        };
        vk_ctx->procs_dev.CmdCopyBuffer(
            s->gpu_resources.command_buffer,
            s->gpu_resources.buffer_positions.buffer,
            s->gpu_resources.buffer_staging_positions.buffer,
            1, // regionCount
            &copy_info_positions
        );

        VkBufferCopy copy_info_velocities {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = s->gpu_resources.buffer_staging_velocities.allocation_info.size,
        };
        vk_ctx->procs_dev.CmdCopyBuffer(
            s->gpu_resources.command_buffer,
            s->gpu_resources.buffer_velocities.buffer,
            s->gpu_resources.buffer_staging_velocities.buffer,
            1, // regionCount
            &copy_info_velocities
        );

        result = vk_ctx->procs_dev.EndCommandBuffer(s->gpu_resources.command_buffer);
        assertVk(result);

        VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = NULL,
            .pWaitDstStageMask = NULL,
            .commandBufferCount = 1,
            .pCommandBuffers = &s->gpu_resources.command_buffer,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = NULL,
        };
        result = vk_ctx->procs_dev.QueueSubmit(vk_ctx->queue, 1, &submit_info, s->gpu_resources.fence);
        assertVk(result);


        result =
            vk_ctx->procs_dev.WaitForFences(vk_ctx->device, 1, &s->gpu_resources.fence, VK_TRUE, UINT64_MAX);
        assertVk(result);

        result = vk_ctx->procs_dev.ResetFences(vk_ctx->device, 1, &s->gpu_resources.fence);
        assertVk(result);
    }

    downloadBufferFromHostVisibleGpuMemory(
        vk_ctx,
        s->particle_count * sizeof(vec4),
        s->gpu_resources.buffer_staging_positions.allocation,
        s->p_positions
    );
    downloadBufferFromHostVisibleGpuMemory(
        vk_ctx,
        s->particle_count * sizeof(vec4),
        s->gpu_resources.buffer_staging_velocities.allocation,
        s->p_velocities
    );
}


extern "C" SimData create(
    const SimParameters* params,
    const VulkanContext* vk_ctx,
    u32fast particle_count,
    const vec4* p_initial_positions
) {

    LOG_F(INFO, "Initializing fluid sim.");

    ZoneScoped;

    SimData s {};
    {
        s.particle_count = particle_count;

        s.p_positions = mallocArray(particle_count, vec4);
        memcpy(s.p_positions, p_initial_positions, particle_count * sizeof(vec4));

        s.p_velocities = callocArray(particle_count, vec4);

        s.p_particles_scratch_buffer1 = callocArray(particle_count, vec4);
        s.p_particles_scratch_buffer2 = callocArray(particle_count, vec4);

        s.cell_count = 0;
        s.C_begin = callocArray(particle_count + 1, u32);
        s.C_length = callocArray(particle_count, u32);

        s.p_cells_scratch_buffer1 = callocArray(particle_count + 1, u32);
        s.p_cells_scratch_buffer2 = callocArray(particle_count + 1, u32);

        s.p_scratch_keyval_buffer_1 = callocArray(particle_count + 1, KeyVal);
        s.p_scratch_keyval_buffer_2 = callocArray(particle_count + 1, KeyVal);

        // smallest prime number larger than the maximum number of particles
        // OPTIMIZE profile this and optimize if too slow
        u32fast hash_modulus = getNextPrimeNumberExclusive(particle_count);
        assert(hash_modulus <= UINT32_MAX);
        s.hash_modulus = (u32)hash_modulus;

        s.H_begin = callocArray(hash_modulus, u32);
        s.H_length = callocArray(hash_modulus, u32);

        setParams(&s, params);

        s.gpu_resources = createGpuResources(vk_ctx, particle_count, hash_modulus);

        uploadDataToGpu(&s, vk_ctx);
        // signal the fence, so that we don't deadlock when waiting for it in `advance()`.
        emptyQueueSubmit(vk_ctx, VK_NULL_HANDLE, VK_NULL_HANDLE, s.gpu_resources.fence);


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

        // TODO Put the thread pool in some central place where it can be accessed from multiple modules.
        s.thread_pool = thread_pool::create(s.processor_count, s.processor_count);
        alwaysAssert(s.thread_pool != NULL);
    }

    LOG_F(
         INFO,
         "Initialized fluid sim with %" PRIuFAST32 " particles, workgroup_size=%u, workgroup_count=%u.",
         s.particle_count, s.gpu_resources.workgroup_size, s.gpu_resources.workgroup_count
     );

    return s;
}


extern "C" void destroy(SimData* s, const VulkanContext* vk_ctx) {

    thread_pool::destroy(s->thread_pool);

    destroyGpuResources(&s->gpu_resources, vk_ctx);

    free(s->p_positions);
    free(s->p_velocities);

    free(s->p_particles_scratch_buffer1);
    free(s->p_particles_scratch_buffer2);

    free(s->p_cells_scratch_buffer1);
    free(s->p_cells_scratch_buffer2);

    free(s->p_scratch_keyval_buffer_1);
    free(s->p_scratch_keyval_buffer_2);

    free(s->C_begin);
    free(s->C_length);

    free(s->H_begin);
    free(s->H_length);

    s->particle_count = 0;
    s->cell_count = 0;
    s->hash_modulus = 0;
}


extern "C" void advance(
    SimData* s,
    const VulkanContext* vk_ctx,
    f32 delta_t,
    VkSemaphore optional_wait_semaphore,
    VkSemaphore optional_signal_semaphore
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

    downloadDataFromGpu(s, vk_ctx);


    const u32fast particle_count = s->particle_count;
    const f32 cell_size_reciprocal = s->parameters.cell_size_reciprocal;

    vec3 domain_min = vec3(INFINITY);
    vec3 domain_max = vec3(-INFINITY);
    {
        ZoneScopedN("compute domain");

        for (u32fast i = 0; i < particle_count; i++)
        {
            domain_min = glm::min(vec3(s->p_positions[i]), domain_min);
            domain_max = glm::max(vec3(s->p_positions[i]), domain_max);
        }
    }

    {
        uvec3 cell_count = uvec3(glm::ceil((domain_max - domain_min) * cell_size_reciprocal) + 0.5f);
        // 32-bit Morton codes can handle at most a 1024x1024x1024 grid.
        // If this turns out to be insufficient, consider using 64-bit Morton codes.
        assert(cell_count.x < 1024 and cell_count.y < 1024 and cell_count.z < 1024);
        (void)cell_count; // to prevent "unused variable" complaints when compiling with NDEBUG
    }

    sortParticles(

        s->thread_pool,
        s->processor_count,

        domain_min,
        cell_size_reciprocal,

        particle_count,

        &s->p_positions,
        &s->p_velocities,

        &s->p_particles_scratch_buffer1,
        &s->p_particles_scratch_buffer2,

        s->p_scratch_keyval_buffer_1,
        s->p_scratch_keyval_buffer_2
    );

    // fill cell list
    {
        ZoneScopedN("fillCellList");

        u32 prev_morton_code = 0;
        if (particle_count > 0)
        {
            s->C_begin[0] = 0;
            prev_morton_code = cellMortonCode(cellIndex(vec3(s->p_positions[0]), domain_min, cell_size_reciprocal));
        }

        u32fast cell_idx = 1;
        for (u32fast particle_idx = 1; particle_idx < particle_count; particle_idx++)
        {
            u32 morton_code =
                cellMortonCode(cellIndex(vec3(s->p_positions[particle_idx]), domain_min, cell_size_reciprocal));

            if (morton_code == prev_morton_code) continue;

            prev_morton_code = morton_code;

            s->C_begin[cell_idx] = (u32)particle_idx;

            cell_idx++;
        }
        s->C_begin[cell_idx] = (u32)particle_count;

        const u32fast cell_count = cell_idx;
        s->cell_count = cell_count;

        for (cell_idx = 0; cell_idx < cell_count; cell_idx++)
        {
            s->C_length[cell_idx] = (u32)s->C_begin[cell_idx+1] - s->C_begin[cell_idx];
        }

        sortCells(
            s->thread_pool,
            s->processor_count,
            s->cell_count,
            &s->C_begin,
            &s->C_length,
            &s->p_cells_scratch_buffer1,
            &s->p_cells_scratch_buffer2,
            s->p_scratch_keyval_buffer_1,
            s->p_scratch_keyval_buffer_2,
            s->p_positions,
            domain_min,
            cell_size_reciprocal,
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
            const u32 particle_idx = s->C_begin[0];
            const vec3 particle = vec3(s->p_positions[particle_idx]);
            const uvec3 cell_idx_3d = cellIndex(particle, domain_min, cell_size_reciprocal);
            const u32 morton_code = cellMortonCode(cell_idx_3d);
            prev_hash = mortonCodeHash(morton_code, s->hash_modulus);
        }
        u32 hash = UINT32_MAX;
        u32 cells_with_this_hash_count = 1;

        for (u32 cell_idx = 1; cell_idx < cell_count; cell_idx++)
        {
            const u32 particle_idx = s->C_begin[cell_idx];
            const vec3 particle = vec3(s->p_positions[particle_idx]);
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


    // OPTIMIZE we can probably wait later than here
    if (optional_wait_semaphore != VK_NULL_HANDLE) {

        ZoneScoped;

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


    result = vk_ctx->procs_dev.ResetCommandBuffer(s->gpu_resources.command_buffer, 0);
    assertVk(result);

    VkCommandBufferBeginInfo begin_info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    result = vk_ctx->procs_dev.BeginCommandBuffer(s->gpu_resources.command_buffer, &begin_info);
    assertVk(result);
    {
        TracyVkZone(vk_ctx->tracy_vk_ctx, s->gpu_resources.command_buffer, "sim");

        // update velocities
        {
            const PushConstants push_constants {
                .domain_min = domain_min,
                .delta_t = delta_t,
                .cell_count = (u32)s->cell_count
            };
            vk_ctx->procs_dev.CmdPushConstants(
                s->gpu_resources.command_buffer,
                s->gpu_resources.pipeline_layout_updateVelocities,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0, // offset
                sizeof(PushConstants),
                &push_constants
            );

            vk_ctx->procs_dev.CmdBindDescriptorSets(
                s->gpu_resources.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                s->gpu_resources.pipeline_layout_updateVelocities,
                0, // firstSet
                1, // descriptorSetCount
                &s->gpu_resources.descriptor_set_main,
                0, // dynamicOffsetCount
                NULL // pDynamicOffsets
            );

            vk_ctx->procs_dev.CmdBindPipeline(
                s->gpu_resources.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                s->gpu_resources.pipeline_updateVelocities
            );

            vk_ctx->procs_dev.CmdDispatch(
                s->gpu_resources.command_buffer,
                s->gpu_resources.workgroup_count, // groupCountX
                1, // groupCountY
                1 // groupCountZ
            );
        }

        {
            VkBufferMemoryBarrier buffer_memory_barrier {
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .srcQueueFamilyIndex = vk_ctx->queue_family_index,
                .dstQueueFamilyIndex = vk_ctx->queue_family_index,
                .buffer = s->gpu_resources.buffer_positions.buffer,
                .offset = 0,
                .size = s->particle_count * sizeof(vec4),
            };
            vk_ctx->procs_dev.CmdPipelineBarrier(
                s->gpu_resources.command_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, // dependencyFlags
                0, // memoryBarrierCount
                NULL, // pMemoryBarriers
                1, // bufferMemoryBarrierCount
                &buffer_memory_barrier, // pBufferMemoryBarriers
                0, // imageMemoryBarrierCount
                NULL // pImageMemoryBarriers
            );
        }

        // update positions
        {
            const PushConstants push_constants {
                .domain_min = domain_min,
                .delta_t = delta_t,
                .cell_count = (u32)s->cell_count
            };
            vk_ctx->procs_dev.CmdPushConstants(
                s->gpu_resources.command_buffer,
                s->gpu_resources.pipeline_layout_updatePositions,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0, // offset
                sizeof(PushConstants),
                &push_constants
            );

            vk_ctx->procs_dev.CmdBindDescriptorSets(
                s->gpu_resources.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                s->gpu_resources.pipeline_layout_updatePositions,
                0, // firstSet
                1, // descriptorSetCount
                &s->gpu_resources.descriptor_set_main,
                0, // dynamicOffsetCount
                NULL // pDynamicOffsets
            );

            vk_ctx->procs_dev.CmdBindPipeline(
                s->gpu_resources.command_buffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                s->gpu_resources.pipeline_updatePositions
            );

            vk_ctx->procs_dev.CmdDispatch(
                s->gpu_resources.command_buffer,
                s->gpu_resources.workgroup_count, // groupCountX
                1, // groupCountY
                1 // groupCountZ
            );
        }

        // @nocompile bind relevant descriptor sets and dispatch the `min` reduction to find the domain min
        // @nocompile After that, download only the Morton codes instead of the positions and velocities
        //     buffers, and send the permutation back up to the GPU.

        TracyVkCollect(vk_ctx->tracy_vk_ctx, s->gpu_resources.command_buffer);
    }
    result = vk_ctx->procs_dev.EndCommandBuffer(s->gpu_resources.command_buffer);
    assertVk(result);

    {
        ZoneScopedN("SubmitCommandBuffer");

        const VkSubmitInfo submit_info {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = NULL,
            .pWaitDstStageMask = 0,
            .commandBufferCount = 1,
            .pCommandBuffers = &s->gpu_resources.command_buffer,
            .signalSemaphoreCount = optional_signal_semaphore == VK_NULL_HANDLE ? (u32)0 : (u32)1,
            .pSignalSemaphores = &optional_signal_semaphore,
        };
        result = vk_ctx->procs_dev.QueueSubmit(vk_ctx->queue, 1, &submit_info, s->gpu_resources.fence);
        assertVk(result);
    }
};


extern "C" void getPositionsVertexBuffer(
    const SimData* s,
    VkBuffer* buffer_out,
    VkDeviceSize* buffer_size_out
) {
    *buffer_out = s->gpu_resources.buffer_positions.buffer;
    *buffer_size_out = s->particle_count * sizeof(vec4);
}

//
// ===========================================================================================================
//

} // namespace
