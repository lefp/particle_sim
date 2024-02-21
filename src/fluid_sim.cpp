#include <cstring>
#include <cstdlib>
#include <cinttypes>
#include <cmath>

#include <glm/glm.hpp>
#include <loguru/loguru.hpp>

#include "types.hpp"
#include "error_util.hpp"
#include "math_util.hpp"
#include "alloc_util.hpp"
#include "fluid_sim.hpp"

namespace fluidsim {

using glm::vec3;

//
// ===========================================================================================================
//

constexpr f32 PI = (f32)M_PI;

constexpr f32 REST_MASS_DENSITY = 1000.0; // kg / m^3
constexpr f32 REST_PARTICLE_DENSITY = 1000.0; // particles / m^3

// The number of particles within the interaction radius at rest. This can be tuned.
constexpr u32fast REST_PARTICLE_INTERACTION_COUNT_APPROX = 50; 

constexpr f32 SPRING_STIFFNESS = 0.05f; // TODO FIXME didn't really think about this



constexpr f32 PARTICLE_MASS = REST_MASS_DENSITY / REST_PARTICLE_DENSITY;

// Number of particles contained in sphere at rest ~= sphere volume * rest particle density.
// :: N = (4/3 pi r^3) rho
// :: r = cuberoot(N * 3 / (4 pi rho)).
const f32 PARTICLE_INTERACTION_RADIUS = cbrtf(
    (f32)REST_PARTICLE_INTERACTION_COUNT_APPROX * 3.f / (4.f * PI * REST_PARTICLE_DENSITY)
);

const f32 SPRING_REST_LENGTH = PARTICLE_INTERACTION_RADIUS / 2.f; // TODO FIXME didn't really think about this

//
// ===========================================================================================================
//

extern SimData init(u32fast particle_count, const vec3* p_initial_positions) {

    SimData s {};

    s.particle_count = particle_count;

    s.p_positions = mallocArray(particle_count, vec3);
    memcpy(s.p_positions, p_initial_positions, particle_count * sizeof(vec3));

    s.p_velocities = callocArray(particle_count, vec3);

    return s;
}


extern void advance(SimData* s, f32 delta_t) {

    assert(delta_t > 1e-5); // assert nonzero

    u32fast particle_count = s->particle_count;
    for (u32fast i = 0; i < particle_count; i++)
    {
        vec3 accel_i = vec3(0);
        vec3 pos_i = s->p_positions[i];

        for (u32fast j = 0; j < particle_count; j++)
        {
            if (i == j) continue;

            vec3 disp = s->p_positions[j] - pos_i;
            f32 dist = glm::length(disp);

            if (dist >= PARTICLE_INTERACTION_RADIUS) continue;
            if (dist < 1e-7)
            {
                LOG_F(WARNING, "distance too small: %" PRIuFAST32 " %" PRIuFAST32 "%f", i, j, dist);
                continue;
            }
            vec3 disp_unit = disp / dist;

            accel_i += SPRING_STIFFNESS * (dist - SPRING_REST_LENGTH) * disp_unit;
        }

        s->p_velocities[i] += accel_i * delta_t;
        s->p_velocities[i] -= 0.5f * delta_t * s->p_velocities[i]; // damping
    }

    for (u32fast i = 0; i < particle_count; i++)
    {
        s->p_positions[i] += s->p_velocities[i] * delta_t;
    }
};

//
// ===========================================================================================================
//

} // namespace
