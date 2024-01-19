#version 450

layout(location = 0) in uint voxel_idx_;

layout(binding = 0, std140) uniform Uniforms {
    mat4 world_to_screen_transform_;
};

struct Voxel {
    ivec3 coord;
    uint material_idx;
};
layout(binding = 1, std430) buffer Voxels {
    Voxel voxels_[];
};


// TODO FIXME make this a specialization constant
const float CUBE_RADIUS = 0.5;

const float LINE_RADIUS = 0.01;

const vec3 CUBE_VERTICES[8] = {
    { -CUBE_RADIUS, -CUBE_RADIUS,  CUBE_RADIUS },
    {  CUBE_RADIUS, -CUBE_RADIUS,  CUBE_RADIUS },
    {  CUBE_RADIUS,  CUBE_RADIUS,  CUBE_RADIUS },
    { -CUBE_RADIUS,  CUBE_RADIUS,  CUBE_RADIUS },
    { -CUBE_RADIUS, -CUBE_RADIUS, -CUBE_RADIUS },
    {  CUBE_RADIUS, -CUBE_RADIUS, -CUBE_RADIUS },
    {  CUBE_RADIUS,  CUBE_RADIUS, -CUBE_RADIUS },
    { -CUBE_RADIUS,  CUBE_RADIUS, -CUBE_RADIUS },
};

struct LineSegmentIndexed {
    uint start;
    uint end;
};

// Indices are into CUBE_VERTICES.
const LineSegmentIndexed LINES[12] = {
    { 0, 4 },
    { 1, 5 },
    { 2, 6 },
    { 3, 7 },
    { 0, 1 },
    { 1, 2 },
    { 2, 3 },
    { 3, 0 },
    { 4, 5 },
    { 5, 6 },
    { 6, 7 },
    { 7, 4 },
};

// x is along the width of the line
// y is along the length of the line
const vec2 BASE_RECTANGLE_VERTICES[6] = {
    // first triangle
    { -1.0,  0.0 },
    {  1.0,  0.0 },
    { -1.0,  1.0 },
    // second triangle
    {  1.0,  0.0 },
    { -1.0,  1.0 },
    {  1.0,  1.0 },
};

void main(void) {

    const Voxel voxel = voxels_[voxel_idx_];
    const vec3 voxel_coord = vec3(voxel.coord);

    const vec2 base_vertex = BASE_RECTANGLE_VERTICES[gl_VertexIndex % 6];

    const LineSegmentIndexed line = LINES[gl_VertexIndex / 6];

    vec4 start_point_world = vec4(voxel_coord + CUBE_VERTICES[line.start], 1.0);
    vec4 end_point_world = vec4(voxel_coord + CUBE_VERTICES[line.end], 1.0);

    vec4 start_point_screen = world_to_screen_transform_ * start_point_world;
    vec4 end_point_screen = world_to_screen_transform_ * end_point_world;

    vec2 vec_along_length = end_point_screen.xy - start_point_screen.xy;
    vec2 vec_along_radius = normalize(vec2(vec_along_length.y, -vec_along_length.x)) * LINE_RADIUS;

    vec2 vertex_pos_2d = start_point_screen.xy;
    vertex_pos_2d += vec_along_length * base_vertex.y;
    vertex_pos_2d += vec_along_radius * base_vertex.x;

    gl_Position = vec4(
        vertex_pos_2d,
        start_point_screen.zw * (1.0 - base_vertex.y) + end_point_screen.zw * (base_vertex.y)
    );
}

