#!/bin/python3


import os
import shutil as sh
import tomllib
import sys
import subprocess as sp

import common


STAGE_DIR = "build/B_compileNonPluginSourcesForPlugins"
if (os.path.exists(STAGE_DIR)):
    sh.rmtree(STAGE_DIR)
os.mkdir(STAGE_DIR)


compilation_processes: list[sp.Popen] = []

lib_names: list[str] = common.getLibNames()
for lib_name in lib_names:

    plugin_src_dir = f"plugins_src/{lib_name}"

    with open(f"{plugin_src_dir}/info.toml") as file:
        s = file.read()
        toml = tomllib.loads(s)

    src_filepaths: list[str] = toml["other_source_files"]
    assert(type(src_filepaths) == list)
    for elem in src_filepaths: assert(type(elem) == str)

    for src_filepath in src_filepaths:

        if (not os.path.isfile(src_filepath)):
            sys.exit(f"Error: File `{src_filepath}` does not exist or is not a file.")

        additional_flags: list[str]
        if src_filepath.startswith('libs'):
            tmp: list[str] | None = common.COMPILE_FLAGS_FOR_LIB_SRC_FILES_USED_BY_PLUGINS.get(src_filepath)
            additional_flags = tmp if (tmp is not None) else []
        else:
            additional_flags = ['-isystem', 'libs']

        compiled_object_filename: str = src_filepath.replace('/', '_') + '.o'
        compile_command: list[str] = (
            [
                'g++', '-c', '-fPIC',
                src_filepath,
                '-o', f'{STAGE_DIR}/{compiled_object_filename}',
            ]
            + common.getCompilerFlag_DNDEBUG()
            + common.getCompilerFlag_g()
            + common.getCompilerFlag_O()
            + additional_flags
        )

        process = sp.Popen(compile_command)
        compilation_processes.append(process)

a_compilation_failed: bool = False
for p in compilation_processes:
    p.communicate()
    if p.returncode != 0: a_compilation_failed = True
if a_compilation_failed:
    print('A compilation failed; aborting build.')
    exit(1);

