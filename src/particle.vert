#version 450

vec2 fullscreen_quad_vertices[6] = {
    { -1.0, -1.0 },
    { -1.0,  1.0 },
    {  1.0,  1.0 },
    //
    { -1.0, -1.0 },
    {  1.0,  1.0 },
    {  1.0, -1.0 },
};

void main(void) {
    // We're not doing `VertexIndex % 6` because you're only supposed to run this with one quad as input.
    vec2 vertex_pos = fullscreen_quad_vertices[gl_VertexIndex];
    gl_Position = vec4(vertex_pos, 0.0, 1.0);
}

