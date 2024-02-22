#!/bin/sh

# TODO FIXME This is a temporary solution. Write a better system for compiling plugins.

# TODO FIXME We will also need a way for the caller to specify whether to use debug or release mode.

g++ \
  -fPIC -shared \
  plugins_src/fluid_sim/fluid_sim.cpp \
  src/error_util.cpp \
  libs/loguru/loguru.cpp \
  -o plugins_build/fluid_sim.so \
  -lc \
  -isystem libs \
  -O3
