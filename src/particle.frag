#version 450

layout(location = 0) out vec4 fragment_color_out_;

layout(binding = 0, std140) uniform Uniforms {
    mat4 world_to_screen_transform_;
};
layout(binding = 2, std430) buffer Particles {
    vec3 particles_[];
};
layout(push_constant, std140) uniform PushConstants {
    mat4 world_to_screen_transform_inverse_;
    vec2 viewport_offset_in_window_;
    vec2 viewport_size_in_window_;

    uint particle_count_;
    float particle_radius_;
    float max_travel_distance_;
};


float getNearestParticleDistance(vec3 center) {

    float min_distance = max_travel_distance_;

    for (uint i = 0; i < particle_count_; i++) {
        float dist = length(particles_[i] - center) - particle_radius_;
        min_distance = min(dist, min_distance);
    }

    return max(min_distance, 0.0f);
}

void main(void) {

    vec2 f = gl_FragCoord.xy;

    // gl_FragCoord is in pixels, starting at the top left of the window, including pixels outside the viewport.
    // Normalize to [-1, 1].
    f -= viewport_offset_in_window_;
    f /= viewport_size_in_window_; // [0, 1]
    f = 2.0*f - 1.0; // [-1, 1]
    // The y-axis flip is done by the transform.

    const vec4 pn = world_to_screen_transform_inverse_ * vec4(f.x, f.y, 0.0, 1.0);
    const vec4 pf = world_to_screen_transform_inverse_ * vec4(f.x, f.y, 1.0, 1.0);
    const vec3 point_on_near_plane = pn.xyz / pn.w;
    const vec3 point_on_far_plane = pf.xyz / pf.w;

    const vec3 direction_unit = normalize(point_on_far_plane - point_on_near_plane);
    const vec3 start_pos = point_on_near_plane;

    vec3 pos = start_pos;
    float dist_traveled = 0;

    uint iteration_count = 0; // @debug
    bool hit = false;

    while (true) {

        const float dist = getNearestParticleDistance(pos);

        if (dist < 1e-5) {
            hit = true;
            break;
        }

        pos += dist * direction_unit;
        dist_traveled += dist;

        if (dist_traveled > max_travel_distance_) break;
        if (iteration_count > 100) break; // @debug
        iteration_count++;
    }

    if (hit) {
        fragment_color_out_ = vec4(0.0, 0.5, 1.0, 1.0);

        vec4 projected_back_into_clip_space = world_to_screen_transform_ * vec4(pos, 1.0);
        gl_FragDepth = projected_back_into_clip_space.z / projected_back_into_clip_space.w;
    }
    else {
        fragment_color_out_ = vec4(0.0, 0.0, 0.0, 1.0);
        gl_FragDepth = 1.1;
    }
}
