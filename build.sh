#!/bin/sh

g++ \
  $(pkg-config --cflags glfw3) \
  $(pkg-config --static --libs glfw3) \
  -g3 \
  -o test \
  src/*.cpp
