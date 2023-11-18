#!/bin/sh

ls src | grep -E '\.(vert|frag|comp)$' | while IFS= read -r src_file ; do
  glslc -o build/"$src_file".spv src/"$src_file"
done

g++ \
  $(pkg-config --cflags glfw3) \
  $(pkg-config --static --libs glfw3) \
  -I libs/loguru \
  -g3 \
  -o build/test \
  libs/loguru/loguru.cpp \
  src/*.cpp
