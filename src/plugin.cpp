#include <dlfcn.h>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <cstdio>

#include <glm/glm.hpp>
#include <loguru/loguru.hpp>

#include "types.hpp"
#include "defer.hpp"
#include "error_util.hpp"
#include "math_util.hpp"
#include "alloc_util.hpp"
#include "plugin.hpp"

#include "../build/A_generatePluginHeaders/plugin_infos.hpp"

namespace plugin {

//
// ===========================================================================================================
//

static void* proc_struct_ptrs_[PluginID_COUNT] {};
static void* dl_handles_[PluginID_COUNT] {};
static u16 lib_versions_[PluginID_COUNT] {};

//
// ===========================================================================================================
//

[[nodiscard]] static bool loadLib(PluginID plugin_id) {

    alwaysAssert(0 <= plugin_id and plugin_id < PluginID_COUNT);

    void* p_procs_struct = proc_struct_ptrs_[plugin_id];
    alwaysAssert(p_procs_struct != NULL); // This function expects the struct to already have been allocated.


    const plugin_infos::PluginReloadInfo* plugin_info = &plugin_infos::PLUGIN_RELOAD_INFOS[plugin_id];


    char* lib_path = NULL;
    {
        int len_without_null = snprintf(NULL, 0, "%s.%i", plugin_info->shared_object_path, lib_versions_[plugin_id]);
        alwaysAssert(len_without_null > 0);

        int buf_size = len_without_null + 1;

        lib_path = (char*)mallocAsserted((size_t)len_without_null);
        len_without_null = snprintf(lib_path, (size_t)buf_size, "%s.%i", plugin_info->shared_object_path, lib_versions_[plugin_id]);
        alwaysAssert(len_without_null < buf_size);
    }
    defer(free(lib_path));


    void* dl_handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);

    if (dl_handle == NULL) {

        const char* err_description = dlerror();
        if (err_description == NULL) err_description = "(NO ERROR DESCRIPTION PROVIDED)";

        LOG_F(
            ERROR, "Failed to load shared library `%s`; dlerror(): `%s`.",
            plugin_info->shared_object_path, err_description
        );

        return false;
    }


    const u32fast proc_count = plugin_info->proc_count;
    for (u32fast proc_idx = 0; proc_idx < proc_count; proc_idx++) {

        const plugin_infos::PluginProcInfo* proc_info = &plugin_info->p_proc_infos[proc_idx];

        const void* p_proc = dlsym(dl_handle, proc_info->proc_name);
        if (p_proc == NULL) {

            const char* err_description = dlerror();
            if (err_description == NULL) err_description = "(NO ERROR DESCRIPTION PROVIDED)";

            LOG_F(
                ERROR, "Failed to load procedure `%s` from shared library `%s`; dlerror(): `%s`.",
                proc_info->proc_name, plugin_info->shared_object_path, err_description
            );

            int result = dlclose(dl_handle);
            alwaysAssert(result == 0);

            return false;
        }

        const void** p_struct_member = (const void**)((uintptr_t)p_procs_struct + proc_info->offset_in_procs_struct);
        *p_struct_member = p_proc;
    }


    dl_handles_[plugin_id] = dl_handle;
    return true;
};

/// Returns a pointer to a struct of pointers.
/// The type of struct depends on the plugin you load.
///     E.g. fluid sim would be FluidSimProcs
extern const void* load(PluginID plugin_id) {

    alwaysAssert(0 <= plugin_id and plugin_id < PluginID_COUNT);

    const plugin_infos::PluginProcStructInfo* proc_struct_info = &plugin_infos::PLUGIN_PROC_STRUCT_INFOS[plugin_id];

    void* p_procs_struct = allocAlignedAsserted(proc_struct_info->alignment, proc_struct_info->size);
    proc_struct_ptrs_[plugin_id] = p_procs_struct;

    bool success = loadLib(plugin_id);
    if (!success) return NULL;

    return p_procs_struct;
}

static bool runCommand(const char* command) {

    errno = 0;
    // TODO replace `system` with something that doesn't involve a shell.
    //     You can't use `popen` because it also spawns a shell.
    //     You probably want:
    //         1. fork
    //         2. execve ; make sure to pass the environment variable ANGAME_RELEASE=1 when appropriate
    //         3. waitpid
    int ret = system(command);

    if (ret == 0) return true;
    else {
        int err = errno;
        const char* err_description = strerror(err);
        if (err_description == NULL) err_description = "(NO ERROR DESCRIPTION PROVIDED)";
        LOG_F(
            ERROR, "Failed to run command `%s`; return code %i, errno %i, strerror(): `%s`.",
            command, ret, err, err_description
        );

        return false;
    }
}

extern bool reload(PluginID plugin_id) {

    alwaysAssert(0 <= plugin_id and plugin_id < PluginID_COUNT);

    if (dl_handles_[plugin_id] == NULL) {
        ABORT_F("reload() called on plugin with id %i, but that plugin wasn't loaded.", plugin_id);
    }


    const plugin_infos::PluginReloadInfo* plugin_info = &plugin_infos::PLUGIN_RELOAD_INFOS[plugin_id];

    // TODO FIXME: temporarily disabled until I reimplement versioning in the compile script
    // lib_versions_[plugin_id]++;

    LOG_F(INFO, "Compiling plugin with ID %i using command `%s`.", plugin_id, plugin_info->compile_script);
    {
        bool success = runCommand(plugin_info->compile_script);
        if (!success) {
            LOG_F(ERROR, "Failed to compile plugin with ID %i.", plugin_id);
            return false;
        }
    }
    LOG_F(INFO, "Linking plugin with ID %i using command `%s`.", plugin_id, plugin_info->link_script);
    {
        bool success = runCommand(plugin_info->link_script);
        if (!success) {
            LOG_F(ERROR, "Failed to link plugin with ID %i.", plugin_id);
            return false;
        }
    }


    int result = dlclose(dl_handles_[plugin_id]);
    alwaysAssert(result == 0);
    dl_handles_[plugin_id] = NULL;

    bool success = loadLib(plugin_id);
    return success;
}

//
// ===========================================================================================================
//

} // namespace
