#!/bin/python3

# NOTE: Use of `ccache` is highly recommended.


from time import time
start_time: float = time()

import subprocess as sp
import os
import shutil as sh
from dataclasses import dataclass
from glob import glob


def runStage(script_path: str) -> None:

    stage_name = os.path.basename(script_path)

    print(f"::::::::::::: Running stage `{stage_name}`")
    t0: float = time()

    result: sp.CompletedProcess = sp.run(script_path)
    result.check_returncode()

    t1: float = time()
    t_total = t1 - t0
    print(f"Completed stage `{stage_name}` (took {t_total:.1} s).")


@dataclass
class FileAndLocation:
    filename: str
    line: int
    col: int


nocompile_list: list[FileAndLocation] = []
src_filepaths: list[str] = (
    glob('src/**', recursive=True) +
    glob('plugins_src/**', recursive=True)
)
assert(len(src_filepaths) > 0)

for filepath in src_filepaths:

    assert(os.path.exists(filepath))
    if (not os.path.isfile(filepath)): continue

    f = open(filepath)
    file_contents: list[str] = f.readlines()
    f.close()

    for line_idx, line_contents in enumerate(file_contents):
        col: int = line_contents.find('@nocompile')
        if (col != -1):
            nocompile_list.append(FileAndLocation(filepath, line_idx, col))

if (len(nocompile_list) != 0):
    print('Error: the following source files contain "@nocompile":')
    for nc in nocompile_list:
        print(f'    - {nc.filename}:{nc.line}:{nc.col}')
    print('Aborting build due to "@nocompile".')
    exit(1)


if (os.path.isdir("build")):
    sh.rmtree("build")
os.mkdir("build")


# TODO whichever stages can be done in parallel, do those in parallel (using Popen)

runStage("build_scripts/A_generatePluginHeaders.py")
runStage("build_scripts/B_compileNonPluginSourcesForPlugins.py")
runStage("build_scripts/C_compilePluginSources.py")
runStage("build_scripts/D_linkPlugins_dependsOn_BC.py")
runStage("build_scripts/E_compileMainProgram_dependsOn_A.py")


sh.copy("build/E_compileMainProgram_dependsOn_A/angame", "build/angame")
sh.copytree("build/E_compileMainProgram_dependsOn_A/shaders", "build/shaders")


end_time: float = time()
total_time: float = end_time - start_time
print(f'{os.path.basename(__file__)}: Done (took {total_time:.1} s).')
