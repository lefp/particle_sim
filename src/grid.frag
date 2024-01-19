#version 450

layout(location = 0) out vec4 fragment_color_out_;

layout(binding = 0, std140) uniform Uniforms {
    mat4 world_to_screen_transform_;
};
layout(push_constant, std140) uniform PushConstants {
    mat4 world_to_screen_transform_inverse_;
    vec2 viewport_offset_in_window_;
    vec2 viewport_size_in_window_;
};

// TODO can make this a push constant or specialization constant
const float grid_interval = 1; // unit: meters
const float gridline_radius = 0.01; // unit: meters. Thickness of gridlines in world coordinates.

// Based on https://asliceofrendering.com/scene%20helper/2020/01/05/InfiniteGrid/
void main(void) {

    vec2 f = gl_FragCoord.xy;

    // gl_FragCoord is in pixels, starting at the top left of the window, including pixels outside the viewport.
    // Normalize to [-1, 1].
    f -= viewport_offset_in_window_;
    f /= viewport_size_in_window_; // [0, 1]
    f = 2.0*f - 1.0; // [-1, 1]
    // The y-axis flip is done by the transform.

    vec4 pn = world_to_screen_transform_inverse_ * vec4(f.x, f.y, 0.0, 1.0);
    vec4 pf = world_to_screen_transform_inverse_ * vec4(f.x, f.y, 1.0, 1.0);
    vec3 point_on_near_plane = pn.xyz / pn.w;
    vec3 point_on_far_plane = pf.xyz / pf.w;

    float lambda = -point_on_near_plane.y / (point_on_far_plane.y - point_on_near_plane.y);
    vec3 point_on_xz_plane = point_on_near_plane + lambda * (point_on_far_plane - point_on_near_plane);

    // compute depth --------------------------------------------------------------------------------------

    vec4 projected_back_into_clip_space = world_to_screen_transform_ * vec4(point_on_xz_plane, 1.0);
    float frag_depth = projected_back_into_clip_space.z / projected_back_into_clip_space.w;

    gl_FragDepth = frag_depth;

    // compute color -----------------------------------------------------------------------------------------

    vec2 pos_on_xz_plane_derivative = fwidth(point_on_xz_plane.xz / grid_interval);
    vec2 grid = abs(fract(point_on_xz_plane.xz / grid_interval - 0.5) - 0.5) / pos_on_xz_plane_derivative;
    float line = min(grid.x, grid.y);
    float minz = min(pos_on_xz_plane_derivative.y, 1);
    float minx = min(pos_on_xz_plane_derivative.x, 1);

    vec4 frag_color = vec4(0.2, 0.2, 0.2, 1.0 - min(line, 1.0));
    if (lambda < 0.0) frag_color = vec4(0.0, 0.0, 0.0, 0.0);

    fragment_color_out_ = frag_color;
}
