#include <cstring>
#include <cstdlib>
#include <cinttypes>
#include <cmath>

#include <glm/glm.hpp>
#include <loguru/loguru.hpp>

#include "../src/types.hpp"
#include "../src/error_util.hpp"
#include "../src/math_util.hpp"
#include "../src/alloc_util.hpp"
#include "fluid_sim_types.hpp"

namespace fluid_sim {

using glm::vec3;

//
// ===========================================================================================================
//

constexpr f32 PI = (f32)M_PI;

//
// ===========================================================================================================
//

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

    LOG_F(INFO, "Set fluid sim parameters: "
        "REST_PARTICLE_DENSITY = %f, "
        "SPRING_STIFFNESS = %f, "
        "SPRING_REST_LENGTH = %f, "
        "PARTICLE_INTERACTION_RADIUS = %f.",
        s->parameters.rest_particle_density,
        s->parameters.spring_stiffness,
        s->parameters.spring_rest_length,
        s->parameters.particle_interaction_radius
    );
}


extern "C" SimData create(const SimParameters* params, u32fast particle_count, const vec3* p_initial_positions) {

    SimData s {};
    {
        s.particle_count = particle_count;

        s.p_positions = mallocArray(particle_count, vec3);
        memcpy(s.p_positions, p_initial_positions, particle_count * sizeof(vec3));

        s.p_velocities = callocArray(particle_count, vec3);

        setParams(&s, params);
    }

    LOG_F(INFO, "Initialized fluid sim with %" PRIuFAST32 " particles.", s.particle_count);

    return s;
}


extern "C" void destroy(SimData* s) {

    free(s->p_positions);
    free(s->p_velocities);
    s->particle_count = 0;
}


extern "C" void advance(SimData* s, f32 delta_t) {

    assert(delta_t > 1e-5); // assert nonzero

    const u32fast particle_count = s->particle_count;
    const f32 particle_interaction_radius = s->parameters.particle_interaction_radius;
    const f32 spring_stiffness = s->parameters.spring_stiffness;
    const f32 spring_rest_length = s->parameters.spring_rest_length;

    for (u32fast i = 0; i < particle_count; i++)
    {
        vec3 accel_i = vec3(0);
        vec3 pos_i = s->p_positions[i];

        for (u32fast j = 0; j < particle_count; j++)
        {
            if (i == j) continue;

            vec3 disp = s->p_positions[j] - pos_i;
            f32 dist = glm::length(disp);

            if (dist >= particle_interaction_radius) continue;
            if (dist < 1e-7)
            {
                LOG_F(WARNING, "distance too small: %" PRIuFAST32 " %" PRIuFAST32 "%f", i, j, dist);
                continue;
            }
            vec3 disp_unit = disp / dist;

            accel_i += spring_stiffness * (dist - spring_rest_length) * disp_unit;
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
