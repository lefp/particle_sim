import os


def isTracyEnabled() -> bool:
    env_str = os.environ.get("ANGAME_NO_TRACY")
    if (env_str is not None and int(env_str) == 1):
        return False
    return True


def getLibNames() -> list[str]:
    files_in_dir: list[str] = os.listdir("plugins_src")
    lib_names: list[str] = []
    for filename in files_in_dir:
        if (os.path.isdir("plugins_src/" + filename)):
            lib_names.append(filename)
    lib_names.sort()

    return lib_names


def getCompilerFlag_O() -> list[str]:
    env_str = os.environ.get("ANGAME_NO_OPTIMIZE")
    if (env_str is not None and int(env_str) == 1):
        return ['-O0']
    return ['-O3']


def getCompilerFlag_DNDEBUG() -> list[str]:
    env_str = os.environ.get("ANGAME_NDEBUG")
    if (env_str is not None and int(env_str) == 1):
        return ['-DNDEBUG']
    return []


def getCompilerFlag_g() -> list[str]:
    env_str = os.environ.get("ANGAME_NO_DEBUG_SYMBOLS")
    if (env_str is not None and int(env_str) == 1):
        return ['-g0']
    return ['-g3']


def getCompilerFlags_TracyDefines() -> list[str]:
    if isTracyEnabled():
        return ['-DTRACY_ENABLE', '-DTRACY_ON_DEMAND', '-DTRACY_NO_BROADCAST']
    else:
        return []


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
