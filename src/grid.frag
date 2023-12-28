#version 450

layout(location = 0) out vec4 fragment_color_out_;

layout(push_constant, std140) uniform PushConstants {
    vec3 camera_direction_unit_;
    vec3 camera_right_direction_unit_;
    vec3 camera_up_direction_unit_;
    vec3 eye_pos_;
    vec2 frustum_near_side_size_;
    float frustum_near_side_distance_;
};

// TODO can make this a push constant or specialization constant
const float grid_interval = 1; // unit: meters
const float gridline_radius = 0.02; // unit: meters. Thickness of gridlines in world coordinates.

void main(void) {

    vec2 f = gl_FragCoord.xy;
    f.y = -f.y; // screen y-axis is down, world y-axis is up

    // Direction of this ray.
    vec3 eye_to_near_plane_vector =
        camera_direction_unit_ * frustum_near_side_distance_ +
        camera_right_direction_unit_ * f.x * 0.5*frustum_near_side_size_.x +
        camera_up_direction_unit_ * f.y * 0.5*frustum_near_side_size_.y;

    float ray_travel_distance_to_near_plane = length(eye_to_near_plane_vector);
    vec3 ray_direction_unit = eye_to_near_plane_vector / ray_travel_distance_to_near_plane;

    // Find intersection with XZ-plane, as follows:
    // Solve
    //     (eye_pos_ + lambda*v).y = 0.
    // Then (eye_pos_ + lambda*v) is the intersection point.
    float v_y = ray_direction_unit.y;
    if (abs(v_y) < 1e-5) v_y = 1; // don't divide by zero
    float lambda = -eye_pos_.y / v_y;

    vec2 pos_on_xz_plane = eye_pos_.xz + lambda*ray_direction_unit.xz;


    float pos_depth = lambda - ray_travel_distance_to_near_plane;


    vec2 pos_mod_grid_interval = mod(pos_on_xz_plane, vec2(grid_interval));
    bool pos_is_on_a_gridline =
        pos_depth >= 0 && (
            any(lessThan(pos_mod_grid_interval, vec2(gridline_radius))) ||
            any(lessThan(vec2(grid_interval) - pos_mod_grid_interval, vec2(gridline_radius)))
        );


    vec4 frag_color;
    if (pos_is_on_a_gridline) frag_color = vec4(0.5, 0.5, 0.5, 1.0);
    else frag_color = vec4(0.0, 0.0, 0.0, 0.0);

    fragment_color_out_ = frag_color;
    // TODO output depth information somewhere
    // Keep in mind that we might want distance from the near plane, not distance from the eye.
}
