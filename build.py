#!/bin/python3


from time import time
start_time: float = time()

import subprocess as sp
import os
import shutil as sh


def runStage(script_path: str) -> None:

    stage_name = os.path.basename(script_path)

    print(f"::::::::::::: Running stage `{stage_name}`")
    t0: float = time()

    result: sp.CompletedProcess = sp.run(script_path)
    result.check_returncode()

    t1: float = time()
    t_total = t1 - t0
    print(f"Completed stage `{stage_name}` (took {t_total:.1} s).")


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
