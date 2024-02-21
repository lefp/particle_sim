#ifndef _FLUID_SIM_HPP
#define _FLUID_SIM_HPP

#include <glm/glm.hpp>
#include "types.hpp"

namespace fluidsim {

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

SimData create(const SimParameters* params, u32fast particle_count, const vec3* p_initial_positions);
void destroy(SimData* sim);

void setParams(SimData*, const SimParameters*);

void advance(SimData*, f32 delta_t);

//
// ===========================================================================================================
//

} // namespace

#endif // include guard
