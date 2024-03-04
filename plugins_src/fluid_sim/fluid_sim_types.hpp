#ifndef _FLUID_SIM_TYPES_HPP
#define _FLUID_SIM_TYPES_HPP

// #include "../../src/types.hpp"
// #include "../../libs/glm/glm.hpp"

namespace fluid_sim {

using glm::vec3;

//
// ===========================================================================================================
//

struct SimParameters {
    f32 rest_particle_density; // particles / m^3
    /// The number of particles within the interaction radius at rest.
    f32 rest_particle_interaction_count_approx;
    f32 spring_stiffness;
};

struct SimData {
    u32fast particle_count;
    vec3* p_positions;
    vec3* p_velocities;

    vec3* p_particles_scratch_buffer1;
    vec3* p_particles_scratch_buffer2;

    u32* p_cells_scratch_buffer1;
    u32* p_cells_scratch_buffer2;

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
    } parameters;
    f32 cell_size; // edge length
    f32 cell_size_reciprocal;
};

//
// ===========================================================================================================
//

}

#endif // include guard
