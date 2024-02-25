#!/bin/python3

import subprocess as sp
import os
import shutil as sh
import subprocess as sp
import sys

import common


STAGE_DIR = "build/D_linkPlugins_dependsOn_BC"
if (os.path.exists(STAGE_DIR)):
    # TODO FIXME you shouldn't do this, because you are deleting all the linked plugins; even the ones you
    # aren't about to relink.
    sh.rmtree(STAGE_DIR)
os.mkdir(STAGE_DIR)


is_release_build: bool = common.isReleaseBuild()


lib_names: list[str]
lib_versions: list[int]
if (len(sys.argv) == 0):
    sys.exit("len(argv) is 0. It's not technically bad but it's weird, maybe something is wrong?")
elif (len(sys.argv) == 1):
    lib_names = common.getLibNames()
    lib_versions = [0] * len(lib_names)
else:

    args = sys.argv[1:]
    lib_names = []
    lib_versions = []

    # Expecting libs in format: `name1 version1 name2 version2 ...`
    #     E.g. `<script_filepath> fluid_sim 4 audio 2`.

    assert(len(args) % 2 == 0)
    for i in range(len(args) // 2):
        name = args[2*i]
        version = (int(args[2*i + 1]))
        assert(version >= 0)

        lib_names.append(name)
        lib_versions.append(version)

    all_libs = common.getLibNames()
    nonexistent_lib: bool = False

    for lib_name in lib_names:
        if not (lib_name in all_libs):
            print(f"Error: No plugin named `{lib_name}`.", file=sys.stderr)
            nonexistent_lib = True
    if (nonexistent_lib): sys.exit("Error: Not all requested plugins exist.")


for (lib_name, lib_version) in zip(lib_names, lib_versions):

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

    shared_object_filename = f'{lib_name}.so.{lib_version}'

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
