#ifndef _FLUID_SIM_TYPES_HPP
#define _FLUID_SIM_TYPES_HPP

// #include <VulkanMemoryAllocator/vk_mem_alloc.h>
// #include "../../src/types.hpp"
// #include "../../libs/glm/glm.hpp"
// #include "../../src/vk_procs.hpp"
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
    VkCommandBuffer general_purpose_command_buffer;
    VkCommandBuffer morton_code_command_buffer;


    VkDescriptorPool descriptor_pool;

    VkDescriptorSetLayout descriptor_set_layout_main;
    VkDescriptorSet descriptor_set_main;

    VkDescriptorSetLayout descriptor_set_layout_reduction;
    VkDescriptorSet descriptor_set_reduction__positions_to_reduction1;
    VkDescriptorSet descriptor_set_reduction__reduction1_to_reduction2;
    VkDescriptorSet descriptor_set_reduction__reduction2_to_reduction1;


    VkPipeline pipeline_updateParticles;
    VkPipelineLayout pipeline_layout_updateParticles;

    VkPipeline pipeline_computeMin;
    VkPipelineLayout pipeline_layout_computeMin;

    VkPipeline pipeline_computeMortonCodes;
    VkPipelineLayout pipeline_layout_computeMortonCodes;

    VkPipeline pipeline_sortParticles;
    VkPipelineLayout pipeline_layout_sortParticles;


    VkSemaphore particle_update_finished_semaphore;
    VkFence fence;


    GpuBuffer buffer_uniforms;

    GpuBuffer buffer_positions_sorted;
    GpuBuffer buffer_velocities_sorted;
    GpuBuffer buffer_positions_unsorted;
    GpuBuffer buffer_velocities_unsorted;

    GpuBuffer buffer_C_begin;
    GpuBuffer buffer_C_length;
    GpuBuffer buffer_H_begin;
    GpuBuffer buffer_H_length;

    GpuBuffer buffer_reduction_1;
    GpuBuffer buffer_reduction_2;

    GpuBuffer buffer_morton_codes_or_permutation;
};

struct SimData {
    u32fast particle_count;

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

    u32* p_morton_codes;
    u32* p_permutation;

    struct Params {
        f32 rest_particle_density;
        f32 particle_interaction_radius;
        f32 spring_rest_length;
        f32 spring_stiffness;
        f32 cell_size; // edge length
        f32 cell_size_reciprocal;
    } parameters;

    GpuResources gpu_resources;

    u32 processor_count;
};

//
// ===========================================================================================================
//

}

#endif // include guard
