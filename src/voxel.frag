#version 450

layout(location = 0) in vec4 color_in_;

layout(location = 0) out vec4 color_out_;

void main(void) {
    color_out_ = color_in_;
}
