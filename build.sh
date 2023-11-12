#!/bin/sh

gcc \
  $(pkg-config --cflags glfw3) \
  $(pkg-config --static --libs glfw3) \
  -o test \
  *.cpp
