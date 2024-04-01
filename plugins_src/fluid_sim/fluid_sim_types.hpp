#ifndef _FLUID_SIM_TYPES_HPP
#define _FLUID_SIM_TYPES_HPP

// #include <VulkanMemoryAllocator/vk_mem_alloc.h>
// #include "../../src/types.hpp"
// #include "../../libs/glm/glm.hpp"
// #include "../../src/vk_procs.hpp"
// #include "../../src/thread_pool.hpp"

namespace fluid_sim {

using glm::vec4;

//
// ===========================================================================================================
//

struct SimParameters {
    f32 rest_particle_density; // particles / m^3
    /// The number of particles within the interaction radius at rest.
    f32 rest_particle_interaction_count_approx;
    f32 spring_stiffness;
};

struct GpuResources {

    u32 workgroup_size;
    u32 workgroup_count;


    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    VkDescriptorSetLayout descriptor_set_layout;

    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;

    VkFence fence;


    VkBuffer buffer_uniforms;
    VmaAllocation allocation_uniforms;
    VmaAllocationInfo allocation_info_uniforms;

    VkBuffer buffer_positions;
    VmaAllocation allocation_positions;
    VmaAllocationInfo allocation_info_positions;

    VkBuffer buffer_staging_positions;
    VmaAllocation allocation_staging_positions;
    VmaAllocationInfo allocation_info_staging_positions;

    VkBuffer buffer_velocities;
    VmaAllocation allocation_velocities;
    VmaAllocationInfo allocation_info_velocities;

    VkBuffer buffer_staging_velocities;
    VmaAllocation allocation_staging_velocities;
    VmaAllocationInfo allocation_info_staging_velocities;

    VkBuffer buffer_C_begin;
    VmaAllocation allocation_C_begin;
    VmaAllocationInfo allocation_info_C_begin;

    VkBuffer buffer_C_length;
    VmaAllocation allocation_C_length;
    VmaAllocationInfo allocation_info_C_length;

    VkBuffer buffer_H_begin;
    VmaAllocation allocation_H_begin;
    VmaAllocationInfo allocation_info_H_begin;

    VkBuffer buffer_H_length;
    VmaAllocation allocation_H_length;
    VmaAllocationInfo allocation_info_H_length;
};

struct SimData {
    u32fast particle_count;
    vec4* p_positions;
    vec4* p_velocities;

    vec4* p_particles_scratch_buffer1;
    vec4* p_particles_scratch_buffer2;

    // TODO FIXME: delete these, you can use the other scratch buffers
    u32* p_cells_scratch_buffer1;
    u32* p_cells_scratch_buffer2;

    u32* p_scratch_u32_buffer_1;
    u32* p_scratch_u32_buffer_2;
    u32* p_scratch_u32_buffer_3;
    u32* p_scratch_u32_buffer_4;

    // This is `\mathbb{C}_compact^begin` in the paper "Multi-Level Memory Structures for Simulating and
    // Rendering Smoothed Particle Hydrodynamics" by Winchenbach and Kolb.
    u32fast cell_count;
    u32* p_cells;
    u32* p_cell_lengths; // `\mathbb{C}_compact^length

    u32 hash_modulus;
    u32* H_begin;
    u32* H_length;

    struct {
        f32 rest_particle_density;
        f32 particle_interaction_radius;
        f32 spring_rest_length;
        f32 spring_stiffness;
        f32 cell_size; // edge length
        f32 cell_size_reciprocal;
    } parameters;

    GpuResources gpu_resources;

    thread_pool::ThreadPool* thread_pool;
};

//
// ===========================================================================================================
//

}

#endif // include guard
