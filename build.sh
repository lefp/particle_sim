#!/bin/sh

ls src | grep -E '\.(vert|frag|comp)$' | while IFS= read -r src_file ; do
  glslc -o build/"$src_file".spv src/"$src_file"
done

g++ \
  -Werror -Wall -Wextra \
  -Walloc-zero -Wcast-qual -Wconversion -Wduplicated-branches -Wduplicated-cond -Wfloat-equal -Wformat=2 \
  -Wformat-signedness -Winit-self -Wlogical-op -Wmissing-declarations -Wshadow -Wswitch-default -Wundef \
  -Wunused-result -Wwrite-strings -Wsign-conversion \
  -Wno-missing-field-initializers \
  $(pkg-config --cflags glfw3) \
  $(pkg-config --static --libs glfw3) \
  -isystem libs \
  -g3 \
  -o build/test \
  libs/loguru/loguru.cpp \
  src/*.cpp
