#version 450

layout(location = 0) in vec3 voxel_coord_worldspace_;
layout(location = 1) in vec4 in_color_;

layout(location = 0) out vec4 out_color_;

layout(binding = 0, std140) uniform Uniforms {
    mat4 world_to_screen_transform_;
};
layout(push_constant, std140) uniform PushConstants {
    float particle_radius_;
};

const vec3 CUBE_VERTICES[8] = {
    { -1.0f, -1.0f,  1.0f },
    {  1.0f, -1.0f,  1.0f },
    { -1.0f,  1.0f,  1.0f },
    {  1.0f,  1.0f,  1.0f },
    { -1.0f, -1.0f, -1.0f },
    {  1.0f, -1.0f, -1.0f },
    { -1.0f,  1.0f, -1.0f },
    {  1.0f,  1.0f, -1.0f },
};

const uint TRIANGLES_VERTEX_INDICES[36] = {
    // back face
    1, 2, 0,
    1, 3, 2,
    // front face
    5, 4, 6,
    5, 6, 7,
    // left face
    0, 2, 6,
    0, 6, 4,
    // right face
    1, 5, 7,
    1, 7, 3,
    // top face
    1, 0, 4,
    1, 4, 5,
    // bottom face
    7, 6, 2,
    7, 2, 3,
};

void main(void) {

    const uint cube_vertex_idx = TRIANGLES_VERTEX_INDICES[gl_VertexIndex];
    vec4 vertex_pos = vec4(particle_radius_ * CUBE_VERTICES[cube_vertex_idx], 1.0);

    vertex_pos.xyz = vertex_pos.xyz + voxel_coord_worldspace_;
    vertex_pos = world_to_screen_transform_ * vertex_pos;

    gl_Position = vertex_pos;
    // out_color_ = vec4((float(cube_vertex_idx) * (1.0f / 7.0f)) * vec3(0.0f, 0.5f, 1.0f), 1.0f);
    out_color_ = in_color_;
}


