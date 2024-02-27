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


float getParticleIntersectionDistance(vec3 ray_origin, vec3 ray_direction_unit, vec3 particle_pos) {

    const float r = particle_radius_;

    vec3 p = particle_pos - ray_origin;
    float projected_dot = dot(ray_direction_unit, p);
    vec3 projected = projected_dot * ray_direction_unit;
    float a = length(projected - p);

    if (a >= particle_radius_ || projected_dot < 0.0f) {
        return (1.0f / 0.0f); // TODO is this guaranteed to produce INF?
    }

    float b = sqrt(r*r - a*a);
    float d = length(projected) - b;

    return d;
}

float getNearestParticleIntersection(vec3 ray_origin, vec3 ray_direction_unit) {

    float min_distance = (1.0f / 0.0f); // TODO is this guaranteed to produce INF?

    for (uint i = 0; i < particle_count_; i++) {
        float dist = getParticleIntersectionDistance(ray_origin, ray_direction_unit, particles_[i]);
        min_distance = min(dist, min_distance);
    }

    return min_distance;
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

    const float dist = getNearestParticleIntersection(start_pos, direction_unit);
    if (dist < 0 || dist > max_travel_distance_) discard;

    fragment_color_out_ = vec4(0.0, 0.5, 1.0, 1.0);

    vec3 intersection_point = start_pos + dist * direction_unit;
    vec4 projected_back_into_clip_space = world_to_screen_transform_ * vec4(intersection_point, 1.0);
    gl_FragDepth = projected_back_into_clip_space.z / projected_back_into_clip_space.w;
}
