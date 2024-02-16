#include <cstring>
#include <cstdlib>
#include <cinttypes>

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

constexpr f32 ACCEL_GRAVITY = 0;
constexpr f32 INTERACTION_RADIUS = 0.25;
constexpr f32 REST_DENSITY = 1.0;
constexpr f32 STIFFNESS = 1.0;
constexpr f32 NEAR_STIFFNESS = 1.0;

//
// ===========================================================================================================
//

extern SimData init(u32fast particle_count, const vec3* p_initial_positions) {

    SimData s {};

    s.particle_count = particle_count;

    s.p_positions = mallocArray(particle_count, vec3);
    memcpy(s.p_positions, p_initial_positions, particle_count * sizeof(vec3));

    s.p_velocities = callocArray(particle_count, vec3);
    s.p_old_positions = callocArray(particle_count, vec3);
    s.scratch_buffer = mallocArray(particle_count, vec3);

    return s;
}

static inline void doubleDensityRelaxation(SimData* s, f32 delta_t) {

    u32fast particle_count = s->particle_count;
    vec3* p_displacements = s->scratch_buffer;

    for (u32fast i = 0; i < particle_count; i++) {

        f32 density = 0;
        f32 near_density = 0;

        for (u32fast j = 0; j < particle_count; j++) {

            if (j == i) continue;

            f32 dist = glm::length(s->p_positions[j] - s->p_positions[i]);
            if (dist >= INTERACTION_RADIUS) continue;

            f32 tmp = 1.0f - dist / INTERACTION_RADIUS;
            density += tmp * tmp;
            near_density += tmp * tmp * tmp;
        }

        f32 pressure = STIFFNESS * (density - REST_DENSITY);
        f32 near_pressure = NEAR_STIFFNESS * near_density;

        vec3 displacement = vec3(0.0f);
        for (u32fast j = 0; j < particle_count; j++) {

            if (j == i) continue;

            vec3 difference = s->p_positions[j] - s->p_positions[i];
            f32 distance = glm::length(difference);
            if (distance >= INTERACTION_RADIUS) continue;

            if (distance < 1e-7) {
                LOG_F(
                    WARNING, "Distance too small (%" PRIuFAST32 ", %" PRIuFAST32 ", %f)",
                    i, j, distance
                );
                continue;
            }
            vec3 difference_unit = difference / distance;

            f32 tmp = 1.0f - distance / INTERACTION_RADIUS;

            displacement +=
                (delta_t*delta_t)
                * (
                    pressure * tmp
                    + near_pressure * (tmp*tmp)
                )
                * difference_unit;

        }
        p_displacements[i] = displacement;
    }

    for (u32fast i = 0; i < particle_count; i++) {
        s->p_positions[i] += p_displacements[i];
    }
}

extern void advance(SimData* s, f32 delta_t) {

    assert(delta_t > 1e-5); // assert nonzero

    u32fast particle_count = s->particle_count;

    for (u32fast i = 0; i < particle_count; i++) {
        // apply gravity
        s->p_velocities[i] += delta_t * ACCEL_GRAVITY;
    }

    // modify velocities with pairwise viscosity impulses
    // TODO FIXME applyViscosity // (Section 5.3)

    // save previous position
    // TODO OPTIMIZE you can do this by swapping the pointers, if you don't need p_positions to also contain
    //     the old positions.
    memcpy(s->p_old_positions, s->p_positions, particle_count * sizeof(vec3));

    for (u32fast i = 0; i < particle_count; i++) {
        // advance to predicted position
        s->p_positions[i] += delta_t * s->p_velocities[i];
     }

    // add and remove springs, change rest lengths
    // TODO FIXME adjustSprings // (Section 5.2)
    // modify positions according to springs,
    // double density relaxation, and collisions
    // TODO FIXME applySpringDisplacements // (Section 5.1)
    doubleDensityRelaxation(s, delta_t); // (Section 4)
    // TODO FIXME resolveCollisions // (Section 6)

    for (u32fast i = 0; i < particle_count; i++) {
        // use previous position to compute next velocity
        s->p_velocities[i] = (s->p_positions[i] - s->p_old_positions[i]) / delta_t;
    }
};

//
// ===========================================================================================================
//

} // namespace
