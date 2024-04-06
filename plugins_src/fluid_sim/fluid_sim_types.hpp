#ifndef _FLUID_SIM_TYPES_HPP
#define _FLUID_SIM_TYPES_HPP

// #include <VulkanMemoryAllocator/vk_mem_alloc.h>
// #include "../../src/types.hpp"
// #include "../../libs/glm/glm.hpp"
// #include "../../src/vk_procs.hpp"
// #include "../../src/thread_pool.hpp"
// #include "../../src/sort.hpp"

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

struct GpuBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo allocation_info;
};

struct GpuResources {

    u32 workgroup_size;
    u32 workgroup_count;


    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkDescriptorPool descriptor_pool;

    VkDescriptorSetLayout descriptor_set_layout_main;
    VkDescriptorSet descriptor_set_main;

    VkDescriptorSetLayout descriptor_set_layout_reduction;
    // @nocompile bind
    VkDescriptorSet descriptor_set_reduction__positions_to_reduction1;
    VkDescriptorSet descriptor_set_reduction__reduction1_to_reduction2;
    VkDescriptorSet descriptor_set_reduction__reduction2_to_reduction1;

    VkPipeline pipeline_updateVelocities;
    VkPipelineLayout pipeline_layout_updateVelocities;

    VkPipeline pipeline_updatePositions;
    VkPipelineLayout pipeline_layout_updatePositions;

    VkFence fence;


    GpuBuffer buffer_uniforms;

    GpuBuffer buffer_positions;
    GpuBuffer buffer_staging_positions;
    GpuBuffer buffer_velocities;
    GpuBuffer buffer_staging_velocities;

    GpuBuffer buffer_C_begin;
    GpuBuffer buffer_C_length;
    GpuBuffer buffer_H_begin;
    GpuBuffer buffer_H_length;

    GpuBuffer buffer_reduction_1;
    GpuBuffer buffer_reduction_2;
};

struct SimData {
    u32fast particle_count;
    vec4* p_positions;
    vec4* p_velocities;

    vec4* p_particles_scratch_buffer1;
    vec4* p_particles_scratch_buffer2;

    u32* p_cells_scratch_buffer1;
    u32* p_cells_scratch_buffer2;

    KeyVal* p_scratch_keyval_buffer_1;
    KeyVal* p_scratch_keyval_buffer_2;

    // From the paper "Multi-Level Memory Structures for Simulating and
    // Rendering Smoothed Particle Hydrodynamics" by Winchenbach and Kolb.
    u32fast cell_count;
    u32* C_begin;
    u32* C_length;

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

    u32 processor_count;
    thread_pool::ThreadPool* thread_pool;
};

//
// ===========================================================================================================
//

}

#endif // include guard
