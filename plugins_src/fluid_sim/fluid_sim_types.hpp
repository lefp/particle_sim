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

    struct {
        f32 rest_particle_density;
        f32 particle_interaction_radius;
        f32 spring_rest_length;
        f32 spring_stiffness;
    } parameters;
};

//
// ===========================================================================================================
//

}

#endif // include guard
