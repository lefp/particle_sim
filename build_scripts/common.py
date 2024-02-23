import os


def getLibNames() -> list[str]:
    files_in_dir: list[str] = os.listdir("plugins_src")
    lib_names: list[str] = []
    for filename in files_in_dir:
        if (os.path.isdir("plugins_src/" + filename)):
            lib_names.append(filename)
    lib_names.sort()

    return lib_names


def isReleaseBuild() -> bool:
    environ_str_release = os.environ.get("ANGAME_RELEASE")
    if environ_str_release is not None and int(environ_str_release) == 1:
        return True
    return False


WARNING_FLAGS: list[str] = [

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


COMPILE_FLAGS_FOR_LIB_SRC_FILES_USED_BY_PLUGINS: dict[str, list[str]] = {
    'libs/loguru/loguru.cpp': ['-I', 'libs/loguru'],
}
