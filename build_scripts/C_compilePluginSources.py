#!/bin/python3

import subprocess as sp
import os
import shutil as sh
import tomllib
import subprocess as sp
import sys

import common


# TODO FIXME:
#     Take the plugin names as a list of args.
#          If empty, do all plugins.
#          Otherwise do the listed plugins (after asserting that they exist).


STAGE_DIR = "build/C_compilePluginSources"
if (os.path.exists(STAGE_DIR)):
    sh.rmtree(STAGE_DIR)
os.mkdir(STAGE_DIR)


is_release_build: bool = common.isReleaseBuild()


lib_names = common.getLibNames()
for lib_name in lib_names:

    plugin_compiled_objects_dir = STAGE_DIR + '/' + lib_name
    os.mkdir(plugin_compiled_objects_dir)

    plugin_src_dir = f"plugins_src/{lib_name}"

    with open(f"{plugin_src_dir}/info.toml") as file:
        s = file.read()
        toml = tomllib.loads(s)

    src_filepaths = toml["plugin_source_files"]
    assert(type(src_filepaths) == list)
    for elem in src_filepaths: assert(type(elem) == str)

    for src_filepath in src_filepaths:

        src_filepath = f"{plugin_src_dir}/{src_filepath}"

        if (not os.path.isfile(src_filepath)):
            sys.exit(f"Error: File `{src_filepath}` does not exist or is not a file.")

        warning_flags: list[str] = common.WARNING_FLAGS.copy()
        # TODO FIXME: temporary. Fix this once you figure out a decent format for the header files which we
        # can parse to extract the signatures, or at least to verify that they match the info.toml (although
        # if we can parse it directly, we should ditch the info.toml).
        if ('-Wmissing-declarations' in warning_flags):
            warning_flags.remove('-Wmissing-declarations')

        compiled_object_filename: str = src_filepath.replace('/', '_') + '.o'
        compile_command: list[str] = (
            [
                'g++', '-c', '-fPIC',
                '-isystem', 'libs',
                src_filepath,
                '-o', f'{plugin_compiled_objects_dir}/{compiled_object_filename}',
            ]
            + warning_flags
            + (['-O3', '-DNDEBUG'] if is_release_build else ['-g3'])
        )

        # TODO OPTIMIZE: do this in parallel for all files, using Popen
        result: sp.CompletedProcess = sp.run(compile_command)
        if (result.returncode != 0): sys.exit(f"Error: Failed to compile file `{src_filepath}`.")
