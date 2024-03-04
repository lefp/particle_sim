#include <cstring>
#include <cstdlib>
#include <cinttypes>
#include <cmath>

#define GLM_FORCE_EXPLICIT_CTOR
#include <glm/glm.hpp>
#include <loguru/loguru.hpp>

#include "../src/types.hpp"
#include "../src/error_util.hpp"
#include "../src/math_util.hpp"
#include "../src/alloc_util.hpp"
#include "fluid_sim_types.hpp"

namespace fluid_sim {

using glm::vec3;
using glm::uvec3;

//
// ===========================================================================================================
//

constexpr f32 PI = (f32)M_PI;

//
// ===========================================================================================================
//

#define SWAP(a, b) \
{ \
    static_assert(typeof(a) == typeof(b)); \
    \
    typeof(a) _tmp = a; \
    a = b; \
    b = _tmp; \
}


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


/// Get the index of the cell that contains the particle.
static inline uvec3 cellIndex(vec3 particle, vec3 domain_min, f32 cell_size_reciprocal) {

    return uvec3((domain_min + particle) * cell_size_reciprocal);
}


static inline u32 cellMortonCode(uvec3 cell_index) {
    return
        ((cell_index.x & 0b0000000001) <<  0) |
        ((cell_index.y & 0b0000000001) <<  1) |
        ((cell_index.z & 0b0000000001) <<  2) |
        ((cell_index.x & 0b0000000010) <<  2) |
        ((cell_index.y & 0b0000000010) <<  3) |
        ((cell_index.z & 0b0000000010) <<  4) |
        ((cell_index.x & 0b0000000100) <<  4) |
        ((cell_index.y & 0b0000000100) <<  5) |
        ((cell_index.z & 0b0000000100) <<  6) |
        ((cell_index.x & 0b0000001000) <<  6) |
        ((cell_index.y & 0b0000001000) <<  7) |
        ((cell_index.z & 0b0000001000) <<  8) |
        ((cell_index.x & 0b0000010000) <<  8) |
        ((cell_index.y & 0b0000010000) <<  9) |
        ((cell_index.z & 0b0000010000) << 10) |
        ((cell_index.x & 0b0000100000) << 10) |
        ((cell_index.y & 0b0000100000) << 11) |
        ((cell_index.z & 0b0000100000) << 12) |
        ((cell_index.x & 0b0001000000) << 12) |
        ((cell_index.y & 0b0001000000) << 13) |
        ((cell_index.z & 0b0001000000) << 14) |
        ((cell_index.x & 0b0010000000) << 14) |
        ((cell_index.y & 0b0010000000) << 15) |
        ((cell_index.z & 0b0010000000) << 16) |
        ((cell_index.x & 0b0100000000) << 16) |
        ((cell_index.y & 0b0100000000) << 17) |
        ((cell_index.z & 0b0100000000) << 18) |
        ((cell_index.x & 0b1000000000) << 18) |
        ((cell_index.y & 0b1000000000) << 19) |
        ((cell_index.z & 0b1000000000) << 20) ;
}


static inline u32 mortonCodeHash(u32 cell_morton_code, u32 hash_modulus) {
    return cell_morton_code % hash_modulus;
}


/// Merge sort the particles by their Morton codes.
/// If the result is written to a scratch buffer, swaps the scratch buffer pointer with the appropriate data
/// buffer pointer.
static void mergeSortByMortonCodes(
    const u64 arr_size,
    vec3 **const pp_positions,
    vec3 **const pp_velocities,
    vec3 **const pp_scratch1,
    vec3 **const pp_scratch2,
    const vec3 domain_min,
    const f32 cell_size_reciprocal
) {

    vec3* pos_arr1 = *pp_positions;
    vec3* pos_arr2 = *pp_scratch1;

    vec3* vel_arr1 = *pp_velocities;
    vec3* vel_arr2 = *pp_scratch2;

    assert(arr_size >= 2);

    for (u64 bucket_size = 1; bucket_size < arr_size; bucket_size *= 2)
    {
        const u64 bucket_count = arr_size / bucket_size;

        for (u64 bucket_idx = 0; bucket_idx < bucket_count; bucket_idx += 2)
        {
            u64 idx_a = (bucket_idx    ) * bucket_size;
            u64 idx_b = (bucket_idx + 1) * bucket_size;
            u64 idx_dst = idx_a;

            const u64 idx_a_max = glm::min(idx_a + bucket_size, arr_size);
            const u64 idx_b_max = glm::min(idx_b + bucket_size, arr_size);
            const u64 idx_dst_max = glm::min(idx_dst + 2*bucket_size, arr_size);

            for (; idx_dst < idx_dst_max; idx_dst++)
            {
                u32 morton_code_a;
                if (idx_a < idx_a_max) {
                    vec3 particle_a = pos_arr1[idx_a];
                    const uvec3 cell_a = cellIndex(particle_a, domain_min, cell_size_reciprocal);
                    morton_code_a = cellMortonCode(cell_a);
                }
                else morton_code_a = UINT32_MAX;

                u32 morton_code_b;
                if (idx_b < idx_b_max) {
                    vec3 particle_b = pos_arr1[idx_b];
                    const uvec3 cell_b = cellIndex(particle_b, domain_min, cell_size_reciprocal);
                    morton_code_b = cellMortonCode(cell_b);
                }
                else morton_code_b = UINT32_MAX;

                if (morton_code_a < morton_code_b) {
                    pos_arr2[idx_dst] = pos_arr1[idx_a];
                    vel_arr2[idx_dst] = vel_arr1[idx_a];
                    idx_a++;
                }
                else {
                    pos_arr2[idx_dst] = pos_arr1[idx_b];
                    vel_arr2[idx_dst] = vel_arr1[idx_b];
                    idx_b++;
                }
            }
        }

        vec3* tmp = pos_arr1;
        pos_arr1 = pos_arr2;
        pos_arr2 = tmp;

        tmp = vel_arr1;
        vel_arr1 = vel_arr2;
        vel_arr2 = tmp;
    }

    *pp_positions = pos_arr1;
    *pp_velocities = vel_arr1;
    *pp_scratch1 = pos_arr2;
    *pp_scratch2 = vel_arr2;
}


static void mergeSortByCellHashes(
    const u64 array_size,
    u32 **const pp_cells,
    u32 **const pp_lengths,
    u32 **const pp_scratch1,
    u32 **const pp_scratch2,
    const vec3 *const p_particles,
    const vec3 domain_min,
    const f32 cell_size_reciprocal,
    const u32 hash_modulus
) {

    u32* cells_arr1 = *pp_cells;
    u32* cells_arr2 = *pp_scratch1;

    u32* lens_arr1 = *pp_lengths;
    u32* lens_arr2 = *pp_scratch2;

    assert(array_size >= 2);

    for (u64 bucket_size = 1; bucket_size < array_size; bucket_size *= 2)
    {
        const u64 bucket_count = array_size / bucket_size;

        for (u64 bucket_idx = 0; bucket_idx < bucket_count; bucket_idx += 2)
        {
            u64 idx_a = (bucket_idx    ) * bucket_size;
            u64 idx_b = (bucket_idx + 1) * bucket_size;
            u64 idx_dst = idx_a;

            const u64 idx_a_max = glm::min(idx_a + bucket_size, array_size);
            const u64 idx_b_max = glm::min(idx_b + bucket_size, array_size);
            const u64 idx_dst_max = glm::min(idx_dst + 2*bucket_size, array_size);

            for (; idx_dst < idx_dst_max; idx_dst++)
            {
                u32 cell_hash_a;
                if (idx_a < idx_a_max) {
                    const u32 particle_idx = cells_arr1[idx_a];
                    const uvec3 cell = cellIndex(p_particles[particle_idx], domain_min, cell_size_reciprocal);
                    const u32 morton_code = cellMortonCode(cell);
                    cell_hash_a = mortonCodeHash(morton_code, hash_modulus);
                }
                else cell_hash_a = UINT32_MAX;

                u32 cell_hash_b;
                if (idx_b < idx_b_max) {
                    const u32 particle_idx = cells_arr1[idx_b];
                    const uvec3 cell = cellIndex(p_particles[particle_idx], domain_min, cell_size_reciprocal);
                    const u32 morton_code = cellMortonCode(cell);
                    cell_hash_b = mortonCodeHash(morton_code, hash_modulus);
                }
                else cell_hash_b = UINT32_MAX;

                if (cell_hash_a < cell_hash_b) {
                    cells_arr2[idx_dst] = cells_arr1[idx_a];
                    lens_arr2[idx_dst] = lens_arr1[idx_a];
                    idx_a++;
                }
                else {
                    cells_arr2[idx_dst] = cells_arr1[idx_b];
                    lens_arr2[idx_dst] = lens_arr1[idx_b];
                    idx_b++;
                }
            }
        }

        u32* tmp = cells_arr1;
        cells_arr1 = cells_arr2;
        cells_arr2 = tmp;

        tmp = lens_arr1;
        lens_arr1 = lens_arr2;
        lens_arr2 = tmp;
    }

    *pp_cells = cells_arr1;
    *pp_lengths = lens_arr1;
    *pp_scratch1 = cells_arr2;
    *pp_scratch2 = lens_arr2;
}


static u32fast getNextPrimeNumberExclusive(u32fast n) {

    if (n == 0 || n == 1) return 2;

    n++; // never return input
    if (!(n & 1)) n++; // if even, make it odd

    // maybe this is 1 too large, that's fine
    const u32fast max = (u32fast)(ceil(sqrt((f64)n)) + 0.5);

    while (true)
    {
        bool prime = true;
        for (u32fast i = 3; i <= max; i++)
        {
            if (n % i == 0)
            {
                prime = false;
                break;
            }
        }
        if (prime) return n;
        n += 2;
    }
}


struct CompactCell {
    u32 first_particle_idx;
    u32 particle_count;
};


static inline CompactCell cell3dToCell(const SimData* s, const uvec3 cell_idx_3d, const vec3 domain_min) {

    const u32 morton_code = cellMortonCode(cell_idx_3d);
    const u32 hash = mortonCodeHash(morton_code, s->hash_modulus);

    const u32 first_cell_with_hash_idx = s->H_begin[hash];
    const u32 n_cells_with_hash = s->H_length[hash];

    if (n_cells_with_hash == 0) return CompactCell { .first_particle_idx = UINT32_MAX, .particle_count = 0 };

    u32 cell_idx = first_cell_with_hash_idx;
    const u32 cell_idx_end = cell_idx + n_cells_with_hash;

    for (; cell_idx < cell_idx_end; cell_idx++)
    {
        const u32 first_particle_in_cell_idx = s->p_cells[cell_idx];
        assert(first_particle_in_cell_idx < s->particle_count);

        const vec3 first_particle_in_cell = s->p_positions[first_particle_in_cell_idx];
        if (
            cellMortonCode(cellIndex(first_particle_in_cell, domain_min, s->cell_size_reciprocal))
            == morton_code
        ) {
            return CompactCell {
                .first_particle_idx = first_particle_in_cell_idx,
                .particle_count = s->p_cell_lengths[cell_idx],
            };
        }
    }

    return CompactCell { .first_particle_idx = UINT32_MAX, .particle_count = 0 };
}


static inline CompactCell particleToCell(const SimData* s, const vec3 particle, const vec3 domain_min) {

    const uvec3 cell_idx_3d = cellIndex(particle, domain_min, s->cell_size_reciprocal);
    return cell3dToCell(s, cell_idx_3d, domain_min);
}


static inline vec3 accelerationDueToParticlesInCell(
    const SimData* s,
    const u32fast target_particle_idx,
    const uvec3 cell_idx_3d,
    const vec3 domain_min
) {

    const CompactCell cell = cell3dToCell(s, cell_idx_3d, domain_min);
    if (cell.particle_count == 0) return vec3(0.0f); // cell doesn't exist

    const vec3 pos = s->p_positions[target_particle_idx];

    vec3 accel {};

    u32fast i = cell.first_particle_idx;
    const u32fast i_end = i + cell.particle_count;

    for (; i < i_end; i++)
    {
        // OPTIMIZE: we can remove this check if we know that none of the particles are the target particle.
        //     E.g. if the particle list comes from a different cell than the target particle.
        if (i == target_particle_idx) continue;

        vec3 disp = s->p_positions[i] - pos;
        f32 dist = glm::length(disp);

        if (dist >= s->parameters.particle_interaction_radius) continue;
        if (dist < 1e-7)
        {
            LOG_F(WARNING, "distance too small: %" PRIuFAST32 " %" PRIuFAST32 " %f", target_particle_idx, i, dist);
            continue;
        }
        vec3 disp_unit = disp / dist;

        accel += s->parameters.spring_stiffness * (dist - s->parameters.spring_rest_length) * disp_unit;
    }

    return accel;
}


extern "C" SimData create(
    const SimParameters* params,
    u32fast particle_count,
    const vec3* p_initial_positions
) {

    SimData s {};
    {
        s.particle_count = particle_count;

        s.p_positions = mallocArray(particle_count, vec3);
        memcpy(s.p_positions, p_initial_positions, particle_count * sizeof(vec3));

        s.p_velocities = callocArray(particle_count, vec3);

        s.p_particles_scratch_buffer1 = callocArray(particle_count, vec3);
        s.p_particles_scratch_buffer2 = callocArray(particle_count, vec3);

        s.cell_count = 0;
        s.p_cells = callocArray(particle_count + 1, u32);
        s.p_cell_lengths = callocArray(particle_count, u32);

        s.p_cells_scratch_buffer1 = callocArray(particle_count + 1, u32);
        s.p_cells_scratch_buffer2 = callocArray(particle_count + 1, u32);

        // smallest prime number larger than the maximum number of particles
        // OPTIMIZE profile this and optimize if too slow
        u32fast hash_modulus = getNextPrimeNumberExclusive(particle_count);
        assert(hash_modulus <= UINT32_MAX);
        s.hash_modulus = (u32)hash_modulus;

        s.H_begin = callocArray(hash_modulus, u32);
        s.H_length = callocArray(hash_modulus, u32);

        setParams(&s, params);

        s.cell_size = 2.0f * s.parameters.particle_interaction_radius;
        s.cell_size_reciprocal = 1.0f / s.cell_size;
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
    const f32 cell_size_reciprocal = s->cell_size_reciprocal;

    vec3 domain_min = vec3(INFINITY);
    vec3 domain_max = vec3(-INFINITY);
    for (u32fast i = 0; i < particle_count; i++)
    {
        domain_min = glm::min(s->p_positions[i], domain_min);
        domain_max = glm::max(s->p_positions[i], domain_max);
    }

    {
        uvec3 cell_count = uvec3(glm::ceil((domain_max - domain_min) * cell_size_reciprocal) + 0.5f);
        // 32-bit Morton codes can handle at most a 1024x1024x1024 grid.
        // If this turns out to be insufficient, consider using 64-bit Morton codes.
        assert(cell_count.x < 1024 and cell_count.y < 1024 and cell_count.z < 1024);
        (void)cell_count; // to prevent "unused variable" complaints when compiling with NDEBUG
    }

    mergeSortByMortonCodes(
        particle_count,
        &s->p_positions, &s->p_velocities, &s->p_particles_scratch_buffer1, &s->p_particles_scratch_buffer2,
        domain_min, cell_size_reciprocal
    );

    // fill cell list
    {
        u32 prev_morton_code = 0;
        if (particle_count > 0)
        {
            s->p_cells[0] = 0;
            prev_morton_code = cellMortonCode(cellIndex(s->p_positions[0], domain_min, cell_size_reciprocal));
        }

        u32fast cell_idx = 1;
        for (u32fast particle_idx = 1; particle_idx < particle_count; particle_idx++)
        {
            u32 morton_code =
                cellMortonCode(cellIndex(s->p_positions[particle_idx], domain_min, cell_size_reciprocal));

            if (morton_code == prev_morton_code) continue;

            prev_morton_code = morton_code;

            s->p_cells[cell_idx] = (u32)particle_idx;

            cell_idx++;
        }
        s->p_cells[cell_idx] = (u32)particle_count;

        const u32fast cell_count = cell_idx;
        s->cell_count = cell_count;

        for (cell_idx = 0; cell_idx < cell_count; cell_idx++)
        {
            s->p_cell_lengths[cell_idx] = (u32)s->p_cells[cell_idx+1] - s->p_cells[cell_idx];
        }

        mergeSortByCellHashes(
            cell_count,
            &s->p_cells, &s->p_cell_lengths, &s->p_cells_scratch_buffer1, &s->p_cells_scratch_buffer2,
            s->p_positions,
            domain_min, cell_size_reciprocal,
            s->hash_modulus
        );
    }

    {
        for (u32fast i = 0; i < s->hash_modulus; i++) s->H_begin[i] = UINT32_MAX;
        for (u32fast i = 0; i < s->hash_modulus; i++) s->H_length[i] = 0;

        const u32 cell_count = (u32)s->cell_count;

        s->H_begin[0] = 0;
        if (cell_count > 0) s->H_length[0] = 1;
        u32 prev_hash;
        {
            const u32 particle_idx = s->p_cells[0];
            const vec3 particle = s->p_positions[particle_idx];
            const uvec3 cell_idx_3d = cellIndex(particle, domain_min, cell_size_reciprocal);
            const u32 morton_code = cellMortonCode(cell_idx_3d);
            prev_hash = mortonCodeHash(morton_code, s->hash_modulus);
        }
        u32 hash = UINT32_MAX;
        u32 cells_with_this_hash_count = 1;

        for (u32 cell_idx = 1; cell_idx < cell_count; cell_idx++)
        {
            const u32 particle_idx = s->p_cells[cell_idx];
            const vec3 particle = s->p_positions[particle_idx];
            const uvec3 cell_idx_3d = cellIndex(particle, domain_min, cell_size_reciprocal);
            const u32 morton_code = cellMortonCode(cell_idx_3d);
            hash = mortonCodeHash(morton_code, s->hash_modulus);

            if (hash != prev_hash)
            {
                s->H_begin[hash] = cell_idx;
                s->H_length[prev_hash] = cells_with_this_hash_count;

                // OPTIMIZE: if `cells_with_this_hash_count == 1`, set `H_begin[prev_hash] = cell_idx-1`.
                // Make sure you account for this embedding in lookups.

                cells_with_this_hash_count = 0;
                prev_hash = hash;
            }
            cells_with_this_hash_count++;
        }
        s->H_length[hash] = cells_with_this_hash_count;
    }

    for (u32fast i = 0; i < particle_count; i++)
    {
        vec3 accel_i = vec3(0);
        vec3 pos_i = s->p_positions[i];

        const uvec3 cell_index_3d = cellIndex(pos_i, domain_min, cell_size_reciprocal);

        // TODO FIXME: verify that the unsigned integer wrapping due to `-1` doesn't break the sim.
        {
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3(-1, -1, -1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3(-1, -1,  0), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3(-1, -1,  1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3(-1,  0, -1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3(-1,  0,  0), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3(-1,  0,  1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3(-1,  1, -1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3(-1,  1,  0), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3(-1,  1,  1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 0, -1, -1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 0, -1,  0), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 0, -1,  1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 0,  0, -1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 0,  0,  0), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 0,  0,  1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 0,  1, -1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 0,  1,  0), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 0,  1,  1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 1, -1, -1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 1, -1,  0), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 1, -1,  1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 1,  0, -1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 1,  0,  0), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 1,  0,  1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 1,  1, -1), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 1,  1,  0), domain_min);
            accel_i += accelerationDueToParticlesInCell(s, i, cell_index_3d + uvec3( 1,  1,  1), domain_min);
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
