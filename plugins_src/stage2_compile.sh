#!/bin/sh

# TODO FIXME This is a temporary solution. Write a better system for compiling plugins.

# TODO FIXME We will also need a way for the caller to specify whether to use debug or release mode.

if [ $# -ne 1 ] ; then exit 1 ; fi

g++ \
  -fPIC -shared \
  plugins_src/fluid_sim/fluid_sim.cpp \
  src/error_util.cpp \
  libs/loguru/loguru.cpp \
  -o $1 \
  -lc \
  -isystem libs \
  -O3
