#!/bin/sh

echo 'building -----------------------------------------------------------------------'

# grep --recursive --line-number 'FIXME' src
# if [ $? -eq 0 ] ; then
#   echo '^ You have unresolved FIXMEs. Aborting build.'
#   exit
# fi


DEBUG_FLAG=-g3
PREPROC_DEFS=-DIMGUI_IMPL_VULKAN_NO_PROTOTYPES


ls -d build &> /dev/null && rm -r build
mkdir -p build
mkdir -p build/intermediate_objects


ls src | grep -E '\.(vert|frag|comp)$' | while IFS= read -r src_file ; do
  glslc -o build/"$src_file".spv src/"$src_file"
done


ls libs/imgui | grep -E '\.cpp$' | while IFS= read -r src_file ; do
  echo "Compiling $src_file"
  g++ \
    $DEBUG_FLAG \
    $PREPROC_DEFS \
    -c \
    libs/imgui/"$src_file" \
    -I libs/imgui \
    -o build/intermediate_objects/"$src_file".o
done

echo 'Compiling libs/imgui/backends/imgui_impl_vulkan.cpp'
g++ \
  $DEBUG_FLAG \
  $PREPROC_DEFS \
  -c \
  -I libs/imgui \
  libs/imgui/backends/imgui_impl_vulkan.cpp \
  -o build/intermediate_objects/imgui_impl_vulkan.cpp.o

echo 'Compiling libs/loguru/loguru.cpp'
g++ \
  $DEBUG_FLAG \
  $PREPROC_DEFS \
  -c \
  -I libs/loguru \
  libs/loguru/loguru.cpp \
  -o build/intermediate_objects/loguru.cpp.o


ls src | grep -E '\.cpp$' | while IFS= read -r src_file ; do
  echo "Compiling $src_file"
  g++ \
    -Werror -Wall -Wextra \
    -Walloc-zero -Wcast-qual -Wconversion -Wduplicated-branches -Wduplicated-cond -Wfloat-equal -Wformat=2 \
    -Wformat-signedness -Winit-self -Wlogical-op -Wmissing-declarations -Wshadow -Wswitch -Wundef \
    -Wunused-result -Wwrite-strings -Wsign-conversion \
    -Wno-missing-field-initializers \
    $DEBUG_FLAG \
    $PREPROC_DEFS \
    -isystem libs \
    -c \
    src/"$src_file" \
    -o build/intermediate_objects/"$src_file".o
done

echo "Linking"
g++ \
    $(pkg-config --static --libs glfw3) \
    build/intermediate_objects/*.o \
    -lc -ldl \
    -o build/test

echo 'done ---------------------------------------------------------------------------'
