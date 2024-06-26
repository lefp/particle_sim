#!/bin/python3

import tomllib
import os
import shutil as sh
from dataclasses import dataclass
import sys

import common


def pascalcase(s_in: str) -> str:

    s_chars = list(s_in)

    s_chars[0] = s_chars[0].upper()
    for i, c in enumerate(s_chars):
        if i != 0 and s_chars[i-1] == '_':
            s_chars[i] = c.upper()

    s = "".join(s_chars)
    s = s.replace('_', '')

    return s


lib_names = common.getLibNames()


GENERATED_HEADERS_DIR = "build/A_generatePluginHeaders"
if (os.path.exists(GENERATED_HEADERS_DIR)):
    sh.rmtree(GENERATED_HEADERS_DIR)
os.mkdir(GENERATED_HEADERS_DIR)


@dataclass
class PluginInfo:
    src_dir: str
    shared_object_path: str
    proc_names: list[str]
    procs_struct_name: str
    lib_name: str
    watch_filepaths: list[str]


def appendNonDirFilesRecursive(list_append_out: list[str], dirpath: str) -> None:

    paths: list[str] = os.listdir(dirpath)
    paths = [dirpath + '/' + p for p in paths]

    for path in paths:
        if (path == '.' or path == '..'): continue

        if (os.path.isfile(path)): list_append_out.append(path)
        elif (os.path.isdir(path)): appendNonDirFilesRecursive(list_append_out, path)
        else: sys.exit(f"Error: `{path}` is not a regular file or directory.")


plugin_infos: list[PluginInfo] = []

for lib_name in lib_names:

    plugin_info_src_dir = f"plugins_src/{lib_name}"

    header_filename = f"plugin_{lib_name}.hpp"
    include_guard_macro = f"_{header_filename.upper().replace('.', '_')}"

    with open(f"{plugin_info_src_dir}/info.toml") as file:
        s = file.read()
        toml = tomllib.loads(s)


    plugin_info_shared_object_path = f"build/D_linkPlugins_dependsOn_BC/{lib_name}.so"


    plugin_info_watch_filepaths: list[str] = []
    appendNonDirFilesRecursive(plugin_info_watch_filepaths, plugin_info_src_dir)


    os.mkdir(f"{GENERATED_HEADERS_DIR}/{lib_name}")

    header = open(f"{GENERATED_HEADERS_DIR}/{lib_name}/{header_filename}", "w")

    # --------------------------------------------------------------------------------------------------------

    header.write(f"//! WARNING: this file is auto-generated (by `{os.path.relpath(__file__)}`).\n")
    header.write("\n")

    header.write(f"#ifndef {include_guard_macro}\n")
    header.write(f"#define {include_guard_macro}\n")
    header.write("\n");

    header.write(f'#include "../../plugins_src/{lib_name}/{lib_name}_types.hpp"\n')
    header.write("\n")

    header.write(f"namespace {lib_name}" " {\n");

    header.write(
"""
//
// ===========================================================================================================
//

"""
    )
    plugin_info_proc_names: list[str] = []
    procs = toml["procedures"]
    for proc in procs:
        ret = proc["return"]
        name = proc["name"]
        args = proc["args"]
        arg_count = len(args)

        plugin_info_proc_names.append(name)

        header.write(f"using FN_{name} = {ret} (")
        for i in range(0, arg_count):

            arg = args[i]

            header.write(f"{arg["type"]}")
            if "name" in arg.keys():
                header.write(f" {arg["name"]}")
            if i != arg_count - 1:
                header.write(", ")

        header.write(");\n")

    header.write("\n")

    struct_name_chars = list(lib_name)
    struct_name_chars[0] = struct_name_chars[0].upper()
    for i, c in enumerate(struct_name_chars):
        if i != 0 and struct_name_chars[i-1] == '_':
            struct_name_chars[i] = c.upper()
    struct_name: str = "".join(struct_name_chars)
    struct_name = struct_name.replace('_', '')
    struct_name += "Procs"

    plugin_info_procs_struct_name = struct_name

    header.write(f"struct {struct_name}" " {\n")
    for proc in procs:
        name = proc["name"]
        header.write(f"    FN_{name}* {name};\n")
    header.write("};\n");

    header.write(
"""
//
// ===========================================================================================================
//

"""
    )

    header.write("} // namespace\n");
    header.write("\n")

    header.write("#endif // include guard\n")

    header.close()


    plugin_info = PluginInfo(
        src_dir=plugin_info_src_dir,
        shared_object_path=plugin_info_shared_object_path,
        proc_names=plugin_info_proc_names,
        procs_struct_name=plugin_info_procs_struct_name,
        lib_name=lib_name,
        watch_filepaths = plugin_info_watch_filepaths
    )
    plugin_infos.append(plugin_info)


with open(f"{GENERATED_HEADERS_DIR}/plugin_infos.hpp", "w") as infos_header:

    infos_header.write(f"//! WARNING: this file is auto-generated (by `{os.path.relpath(__file__)}`).\n")


    infos_header.write(
"""
#ifndef _PLUGIN_INFOS_HPP
#define _PLUGIN_INFOS_HPP

#include "plugin_ids.hpp"
"""
    )

    for plugin_info in plugin_infos:
        infos_header.write(f'#include "{plugin_info.lib_name}/plugin_{plugin_info.lib_name}.hpp"\n')

    infos_header.write(
"""
namespace plugin_infos {

//
// ===========================================================================================================
//

struct PluginProcStructInfo {
    u32fast alignment;
    u32fast size;
};

struct PluginProcInfo {
    const char* proc_name;
    u32fast offset_in_procs_struct;
};

struct PluginReloadInfo {

    const char* shared_object_path;

    u32fast proc_count;
    const PluginProcInfo* p_proc_infos;

    const char* compile_script;
    const char* link_script;

    const char* name;

    u32fast watch_filepath_count;
    const char *const *p_watch_filepaths;
};

//
// ===========================================================================================================
//

"""
    )

    plugin_count = len(plugin_infos)
    for plugin_idx, plugin_info in enumerate(plugin_infos):

        infos_header.write(f"constexpr u32fast PROC_COUNT_{lib_name.upper()} = {len(plugin_info.proc_names)};\n")
        infos_header.write(f"constexpr PluginProcInfo PROC_INFOS_{lib_name.upper()}[PROC_COUNT_{plugin_info.lib_name.upper()}]" " {\n")

        for name in plugin_info.proc_names:
            infos_header.write("    PluginProcInfo {\n")
            infos_header.write(f'        .proc_name = "{name}",\n')
            infos_header.write(f'        .offset_in_procs_struct = offsetof({plugin_info.lib_name}::{plugin_info.procs_struct_name}, {name}),\n')
            infos_header.write("    },\n")

        infos_header.write("};\n")

        if (plugin_idx != plugin_count - 1): infos_header.write("\n")

    infos_header.write(
"""
//
// ===========================================================================================================
//

"""
    )

    plugin_count = len(plugin_infos)
    for plugin_idx, plugin_info in enumerate(plugin_infos):

        infos_header.write(f"constexpr u32fast WATCH_FILEPATH_COUNT_{lib_name.upper()} = {len(plugin_info.watch_filepaths)};\n")
        infos_header.write(f"const char *const WATCH_FILEPATHS_{lib_name.upper()}[WATCH_FILEPATH_COUNT_{plugin_info.lib_name.upper()}]" " {\n")

        for path in plugin_info.watch_filepaths:
            infos_header.write(f'    "{path}",\n')

        infos_header.write("};\n")

        if (plugin_idx != plugin_count - 1): infos_header.write("\n")

    infos_header.write(
"""
//
// ===========================================================================================================
//

"""
    )

    infos_header.write("constexpr PluginReloadInfo PLUGIN_RELOAD_INFOS[PluginID_COUNT] {\n")

    for plugin_idx, plugin_info in enumerate(plugin_infos):
        infos_header.write("    PluginReloadInfo {\n")
        infos_header.write(f'        .shared_object_path = "{plugin_info.shared_object_path}",\n')
        infos_header.write(f'        .proc_count = PROC_COUNT_{plugin_info.lib_name.upper()},\n')
        infos_header.write(f'        .p_proc_infos = PROC_INFOS_{plugin_info.lib_name.upper()},\n')
        infos_header.write(f'        .compile_script = "build_scripts/C_compilePluginSources.py",\n')
        infos_header.write(f'        .link_script = "build_scripts/D_linkPlugins_dependsOn_BC.py",\n')
        infos_header.write(f'        .name = "{lib_name}",\n')
        infos_header.write(f'        .watch_filepath_count = WATCH_FILEPATH_COUNT_{lib_name.upper()},\n')
        infos_header.write(f'        .p_watch_filepaths = WATCH_FILEPATHS_{lib_name.upper()},\n')
        infos_header.write("    },\n")

    infos_header.write("};\n")


    infos_header.write("constexpr PluginProcStructInfo PLUGIN_PROC_STRUCT_INFOS[PluginID_COUNT] {\n")

    for plugin_idx, plugin_info in enumerate(plugin_infos):
        infos_header.write("    PluginProcStructInfo {\n")
        infos_header.write(f"        .alignment = alignof({plugin_info.lib_name}::{plugin_info.procs_struct_name}),\n")
        infos_header.write(f"        .size = sizeof({plugin_info.lib_name}::{plugin_info.procs_struct_name}),\n")
        infos_header.write("    },\n")

    infos_header.write("};\n")


    infos_header.write(
"""
//
// ===========================================================================================================
//

} // namespace

#endif // include guard
"""
    )


with open(f"{GENERATED_HEADERS_DIR}/plugin_ids.hpp", "w") as ids_header:
    ids_header.write(
"""
#ifndef _PLUGIN_IDS
#define _PLUGIN_IDS

//
// ===========================================================================================================
//

"""
    )

    ids_header.write("enum PluginID {\n")
    for i, plugin_info in enumerate(plugin_infos):
        ids_header.write(f"    PluginID_{pascalcase(plugin_info.lib_name)} = {i},\n")
    ids_header.write("    PluginID_COUNT\n")
    ids_header.write("};\n")

    ids_header.write(
"""
//
// ===========================================================================================================
//

#endif // include guard
"""
    )
