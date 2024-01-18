#version 450

layout(location = 0) in vec3 in_line_segment_start_point_;
layout(location = 1) in vec3 in_line_segment_end_point_;

layout(binding = 0, std140) uniform Uniforms {
    mat4 world_to_screen_transform_;
};


const float LINE_RADIUS = 0.01;

// x is along the width of the line
// y is along the length of the line
const vec2 BASE_RECTANGLE_VERTICES[4] = {
    { -1.0,  0.0 },
    {  1.0,  0.0 },
    { -1.0,  1.0 },
    {  1.0,  1.0 },
};

const uint TRIANGLES_VERTEX_INDICES[6] = {
    0, 2, 1,
    1, 2, 3,
};

void main(void) {

    const uint cube_vertex_idx = TRIANGLES_VERTEX_INDICES[gl_VertexIndex % 6];
    vec2 base_vertex = BASE_RECTANGLE_VERTICES[cube_vertex_idx];

    vec4 start_point = world_to_screen_transform_ * vec4(in_line_segment_start_point_, 1.0);
    vec4 end_point = world_to_screen_transform_ * vec4(in_line_segment_end_point_, 1.0);

    vec2 vec_along_length = end_point.xy - start_point.xy;
    vec2 vec_along_radius = normalize(vec2(vec_along_length.y, -vec_along_length.x)) * LINE_RADIUS;


    vec2 vertex_pos_2d = start_point.xy;
    vertex_pos_2d += vec_along_length * base_vertex.y;
    vertex_pos_2d += vec_along_radius * base_vertex.x;

    gl_Position = vec4(
        vertex_pos_2d,
        start_point.zw * (1.0 - base_vertex.y) + end_point.zw * (base_vertex.y)
    );
}


