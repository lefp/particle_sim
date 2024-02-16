#ifndef _FLUID_SIM_HPP
#define _FLUID_SIM_HPP

#include <glm/glm.hpp>
#include "types.hpp"

namespace fluidsim {

using glm::vec3;

//
// ===========================================================================================================
//

struct SimData {
    u32fast particle_count;
    vec3* p_positions;
    vec3* p_velocities;
    vec3* p_old_positions;
    vec3* scratch_buffer;
};

SimData init(u32fast particle_count, const vec3* p_initial_positions);

void advance(SimData*, f32 delta_t);

//
// ===========================================================================================================
//

} // namespace

#endif // include guard
