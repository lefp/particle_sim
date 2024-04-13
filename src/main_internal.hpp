#ifndef _MAIN_INTERNAL_HPP
#define _MAIN_INTERNAL_HPP

// #include <glm/glm.hpp>
// #include "types.hpp"

using glm::ivec3;
using glm::vec3;

//
// ===========================================================================================================
//

struct VoxelPosAndIndex {
    ivec3 pos;
    u32 idx;
};


struct Hexahedron {
    // TODO Rename the member variables; they were originally named with a frustum in mind, but a hexahedron
    // is not necessarily a frustum.

    vec3 near_bot_left_p;
    vec3 far_top_right_p;

    vec3 near_normal;
    vec3 bot_normal;
    vec3 left_normal;

    vec3 far_normal;
    vec3 top_normal;
    vec3 right_normal;
};


struct FrustumCullThreadArgs {
    const Hexahedron* frustum;
    const graphics::Voxel* p_voxels;
    VoxelPosAndIndex* p_voxels_out;
    u32 start_idx;
    u32 voxel_count;
    u32 voxels_in_frustum_count_ret;
};

//
// ===========================================================================================================
//

#endif // include guard
