#version 450

vec2 VERTICES[3] = {
    {  0.0, -0.5 },
    { -0.5,  0.5 },
    {  0.5,  0.5 },
};

void main(void) {
    vec2 pos = VERTICES[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);
}
