#version 450

layout(location = 0) in vec4 color_in_;

layout(location = 0) out vec4 color_out_;

void main(void) {
    color_out_ = color_in_;
    float frag_depth = 1.0 - gl_FragCoord.z;
    gl_FragDepth = frag_depth; // TODO FIXME temporary workaround; figure out why it's backwards and fix it
}
