#!/bin/python3

import os
import shutil as sh
import subprocess as sp
import re
from dataclasses import dataclass

import common


debug_build = not common.isReleaseBuild()

TRACY = True

BUILD_DIR_PATH = 'build/E_compileMainProgram_dependsOn_A'
INTERMEDIATE_OBJECTS_PATH = BUILD_DIR_PATH + '/intermediate_objects'
SPIRV_DIR_PATH = BUILD_DIR_PATH + '/shaders'

COMMON_COMPILE_FLAGS: list[str] = (
    (['-g3'] if debug_build else ['-O3', '-DNDEBUG']) +
    ['-DIMGUI_IMPL_VULKAN_NO_PROTOTYPES'] +
    (['-DTRACY_ENABLE', '-DTRACY_ON_DEMAND', '-DTRACY_NO_BROADCAST'] if TRACY else [])
)

glfw_link_flags_str, stderr = (
    sp.Popen(['pkg-config', '--static', '--libs', 'glfw3'], stdout=sp.PIPE, stderr=sp.PIPE).communicate()
)
if (stderr != b''):
    print(stderr.decode())
    os.abort()
glfw_link_flags: list[str] = glfw_link_flags_str.decode().strip().split(' ')
LINK_FLAGS = ['-lc', '-ldl'] + glfw_link_flags


@dataclass
class Library:
    source_file_paths: list[str]
    additional_compile_flags: list[str]

@dataclass
class FileAndLocation:
    filename: str
    line: int
    col: int


def filesInDirWithSuffix(dir: str, suffix: str):
    all_files: list[str] = os.listdir(dir)
    return list(filter(lambda s: s.endswith(suffix), all_files))


nocompile_list: list[FileAndLocation] = []
for filename in os.listdir('src'):

    f = open('src/' + filename)
    file_contents: list[str] = f.readlines()
    f.close()

    for line_idx, line_contents in enumerate(file_contents):
        col: int = line_contents.find('@nocompile')
        if (col != -1):
            nocompile_list.append(FileAndLocation(filename, line_idx, col))

if (len(nocompile_list) != 0):
    print('Error: the following source files contain "@nocompile":')
    for nc in nocompile_list:
        print(f'    - {nc.filename}:{nc.line}:{nc.col}')
    print('Aborting build due to "@nocompile".')
    exit(1)


compilation_processes: list[sp.Popen] = []


os.mkdir(BUILD_DIR_PATH)
os.mkdir(INTERMEDIATE_OBJECTS_PATH)
os.mkdir(SPIRV_DIR_PATH)

# compile shaders --------------------------------------------------------------------------------------------

shader_source_files = filter(
    lambda s: s.endswith('.vert') or s.endswith('.frag') or s.endswith('.comp'),
    os.listdir('src')
)

for shader_source_file in shader_source_files:
    glslc_process = sp.Popen([
        'glslc',
        'src/' + shader_source_file,
        '-o', SPIRV_DIR_PATH + '/' + shader_source_file + '.spv'
    ])
    compilation_processes.append(glslc_process)

# compile cpp files ------------------------------------------------------------------------------------------

lib_my_app = Library(
    source_file_paths = [
        'src/' + s
        for s in filesInDirWithSuffix('src', '.cpp')
    ],
    additional_compile_flags = ['-isystem', 'libs'] + common.WARNING_FLAGS
                               + ['-isystem', 'libs/imgui'] # implot.h #includes "imgui.h", not "libs/imgui.h"
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
    additional_compile_flags = []
)
lib_implot = Library(
    source_file_paths = [
        'libs/implot/' + s
        for s in filesInDirWithSuffix('libs/implot', '.cpp')
    ],
    additional_compile_flags = ['-isystem', 'libs/imgui']
)
lib_tracy = Library(
    source_file_paths = [
        'libs/tracy/' + s
        for s in filesInDirWithSuffix('libs/tracy', '.cpp')
    ],
    additional_compile_flags = ['-isystem', 'libs/tracy']
)

libs = [
    lib_my_app,
    lib_loguru,
    lib_imgui,
    lib_implot,
]
if TRACY: libs.append(lib_tracy)

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
        compilation_processes.append(gcc_process)

a_compilation_failed: bool = False
for p in compilation_processes:
    p.communicate()
    if p.returncode != 0: a_compilation_failed = True

if a_compilation_failed:
    print('A compilation failed; aborting build.')
    exit(1);

# link -------------------------------------------------------------------------------------------------------

gcc_process = sp.Popen(
    [
        'g++',
        '-o', BUILD_DIR_PATH + '/angame',
    ]
    + LINK_FLAGS
    + [
        INTERMEDIATE_OBJECTS_PATH + '/' + s
        for s in filesInDirWithSuffix(INTERMEDIATE_OBJECTS_PATH, '.o')
    ]
)
gcc_process.communicate()
if gcc_process.returncode != 0:
    print('Link failed; aborting build.')
    exit(1)
