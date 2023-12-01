#version 450

layout(location = 0) out vec4 fragment_color_;

layout(push_constant, std140) uniform PushConstants {
    float angle_radians_;
};

const float CUBE_RADIUS = 0.2;

const vec3 CUBE_VERTICES[8] = {
    { -CUBE_RADIUS, -CUBE_RADIUS,  CUBE_RADIUS },
    {  CUBE_RADIUS, -CUBE_RADIUS,  CUBE_RADIUS },
    { -CUBE_RADIUS,  CUBE_RADIUS,  CUBE_RADIUS },
    {  CUBE_RADIUS,  CUBE_RADIUS,  CUBE_RADIUS },
    { -CUBE_RADIUS, -CUBE_RADIUS, -CUBE_RADIUS },
    {  CUBE_RADIUS, -CUBE_RADIUS, -CUBE_RADIUS },
    { -CUBE_RADIUS,  CUBE_RADIUS, -CUBE_RADIUS },
    {  CUBE_RADIUS,  CUBE_RADIUS, -CUBE_RADIUS },
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
    vec3 vertex_pos = CUBE_VERTICES[cube_vertex_idx];

    fragment_color_ = vec4((vertex_pos + CUBE_RADIUS) / (2*CUBE_RADIUS), 1.0);

    mat3 transform;
    {
        vec3 rot_axis = normalize(vec3(1.0, 1.0, 1.0));
        float rx = rot_axis.x;
        float ry = rot_axis.y;
        float rz = rot_axis.z;

        float cos_ang = cos(angle_radians_);
        float sin_ang = sin(angle_radians_);

        // the constructor's parameter order is column-major
        transform = mat3(
            cos_ang + rx*rx * (1 - cos_ang), ry*rx * (1 - cos_ang) + rz*sin_ang, rz*rx * (1 - cos_ang) - ry*sin_ang,
            rx*ry * (1 - cos_ang) - rz*sin_ang, cos_ang + ry*ry * (1 - cos_ang), rz*ry * (1 - cos_ang) + rx*sin_ang,
            rx*rz * (1 - cos_ang) + ry*sin_ang, ry*rz * (1 - cos_ang) - rx*sin_ang, cos_ang + rz*rz * (1 - cos_ang)
        );
    }

    vertex_pos = transform * vertex_pos;


    // Vulkan depth is in range [0, 1]
    vertex_pos.z += 1.0;
    vertex_pos.z *= 0.5;
    gl_Position = vec4(vertex_pos, 1.0);
}

