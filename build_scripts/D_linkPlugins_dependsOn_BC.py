#!/bin/python3

import subprocess as sp
import os
import shutil as sh
import subprocess as sp
import sys

import common


# TODO FIXME:
#     Take the plugin names as a list of args.
#          If empty, do all plugins.
#          Otherwise do the listed plugins (after asserting that they exist).
#     Also, take an optional -o argument.


STAGE_DIR = "build/D_linkPlugins_dependsOn_BC"
if (os.path.exists(STAGE_DIR)):
    sh.rmtree(STAGE_DIR)
os.mkdir(STAGE_DIR)


is_release_build: bool = common.isReleaseBuild()


lib_names: list[str]
if (len(sys.argv) == 0):
    sys.exit("len(argv) is 0. It's not technically bad but it's weird, maybe something is wrong?")
elif (len(sys.argv) == 1):
    lib_names = common.getLibNames()
else:
    lib_names = sys.argv[1:]

    all_libs = common.getLibNames()
    nonexistent_lib: bool = False

    for lib_name in lib_names:
        if not (lib_name in all_libs):
            print(f"Error: No plugin named `{lib_name}`.", file=sys.stderr)
            nonexistent_lib = True
    if (nonexistent_lib): sys.exit("Error: Not all requested plugins exist.")


for lib_name in lib_names:

    plugin_objects_dir = f"build/C_compilePluginSources/{lib_name}"
    other_objects_dir = f"build/B_compileNonPluginSourcesForPlugins"

    plugin_object_filenames = os.listdir(plugin_objects_dir)
    other_object_filenames = os.listdir(other_objects_dir)

    plugin_object_filepaths = [f'{plugin_objects_dir}/{fname}' for fname in plugin_object_filenames]
    other_object_filepaths = [f'{other_objects_dir}/{fname}' for fname in other_object_filenames]

    object_filepaths = plugin_object_filepaths + other_object_filepaths

    for fpath in object_filepaths:
        assert(fpath[-2:] == '.o')
        assert(os.path.isfile(fpath))

    shared_object_filename = f'{lib_name}.so.0'

    link_command = (
        [
            'g++', '-fPIC', '-shared',
            '-o', f'{STAGE_DIR}/{shared_object_filename}',
        ]
        + object_filepaths
    )

    # TODO OPTIMIZE: do this in parallel for all files, using Popen
    result: sp.CompletedProcess = sp.run(link_command)
    if (result.returncode != 0): sys.exit(f"Error: Failed to link file `{shared_object_filename}`.")
