#!/bin/python3

from time import time
start_time: float = time()

import os
import shutil as sh
import subprocess as sp
import re
from dataclasses import dataclass

DEBUG_BUILD = True

BUILD_DIR_PATH = 'build'
INTERMEDIATE_OBJECTS_PATH = BUILD_DIR_PATH + '/intermediate_objects'

COMMON_COMPILE_FLAGS: list[str] = (
    (['-g3'] if DEBUG_BUILD else []) +
    ['-DIMGUI_IMPL_VULKAN_NO_PROTOTYPES']
)

glfw_link_flags_str, stderr = (
    sp.Popen(['pkg-config', '--static', '--libs', 'glfw3'], stdout=sp.PIPE, stderr=sp.PIPE).communicate()
)
if (stderr != b''):
    print(stderr.decode())
    os.abort()
glfw_link_flags: list[str] = glfw_link_flags_str.decode().strip().split(' ')
LINK_FLAGS = ['-lc', '-ldl'] + glfw_link_flags

MY_COMPILE_FLAGS_W = [

    '-Werror',
    '-Wall',
    '-Wextra',

    '-Walloc-zero',
    '-Wcast-qual',
    '-Wconversion',
    '-Wduplicated-branches',
    '-Wduplicated-cond',
    '-Wfloat-equal',
    '-Wformat=2',
    '-Wformat-signedness',
    '-Winit-self',
    '-Wlogical-op',
    '-Wmissing-declarations',
    '-Wshadow',
    '-Wswitch',
    '-Wundef',
    '-Wunused-result',
    '-Wwrite-strings',
    '-Wsign-conversion',

    '-Wno-missing-field-initializers',
]


@dataclass
class Library:
    source_file_paths: list[str]
    additional_compile_flags: list[str]


def filesInDirWithSuffix(dir: str, suffix: str):
    all_files: list[str] = os.listdir(dir)
    return list(filter(lambda s: s.endswith(suffix), all_files))


print('building...');


if os.path.isdir(BUILD_DIR_PATH):
    sh.rmtree(BUILD_DIR_PATH)

os.mkdir(BUILD_DIR_PATH)
os.mkdir(INTERMEDIATE_OBJECTS_PATH)


shader_source_files = filter(
    lambda s: s.endswith('.vert') or s.endswith('.frag') or s.endswith('.comp'),
    os.listdir('src')
)

for shader_source_file in shader_source_files:
    glslc_process = sp.Popen([
        'glslc',
        'src/' + shader_source_file,
        '-o', BUILD_DIR_PATH + '/' + shader_source_file + '.spv'
    ])
    glslc_process.communicate()


lib_my_app = Library(
    source_file_paths = [
        'src/' + s
        for s in filesInDirWithSuffix('src', '.cpp')
    ],
    additional_compile_flags = ['-isystem', 'libs'] + MY_COMPILE_FLAGS_W
)
lib_loguru = Library(
    source_file_paths = ['libs/loguru/loguru.cpp'],
    additional_compile_flags = ['-I', 'libs/loguru']
)
lib_imgui = Library(
    source_file_paths = [
        'libs/imgui/' + s
        for s in filesInDirWithSuffix('libs/imgui', '.cpp')
    ],
    additional_compile_flags = ['-I', 'libs/imgui']
)

libs = [lib_my_app, lib_loguru, lib_imgui]
for lib in libs:
    for source_file_path in lib.source_file_paths:
        gcc_process = sp.Popen(
            [
                'g++',
                '-c',
                '-o', INTERMEDIATE_OBJECTS_PATH + '/' + source_file_path.replace('/', '_') + '.o',
                source_file_path,
            ]
            + COMMON_COMPILE_FLAGS
            + lib.additional_compile_flags
        )
        gcc_process.communicate()

gcc_process = sp.Popen(
    [
        'g++',
        '-o', 'build/test',
    ]
    + LINK_FLAGS
    + [
        INTERMEDIATE_OBJECTS_PATH + '/' + s
        for s in filesInDirWithSuffix(INTERMEDIATE_OBJECTS_PATH, '.o')
    ]
)
gcc_process.communicate()


end_time: float = time()
total_time: float = end_time - start_time
print(f'done ({total_time:.1} s)')

