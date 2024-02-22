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
#include "../plugins_generated/plugin_infos.hpp"
#include "plugin.hpp"

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

extern bool reload(PluginID plugin_id) {

    alwaysAssert(0 <= plugin_id and plugin_id < PluginID_COUNT);


    if (dl_handles_[plugin_id] == NULL) {
        ABORT_F("Reload called on plugin with id %i, but that plugin wasn't loaded.", plugin_id);
    }

    int result = dlclose(dl_handles_[plugin_id]);
    alwaysAssert(result == 0);

    dl_handles_[plugin_id] = NULL;


    const plugin_infos::PluginReloadInfo* plugin_info = &plugin_infos::PLUGIN_RELOAD_INFOS[plugin_id];

    // TODO FIXME we should dlclose() the previously-loaded version and free the procs struct

    lib_versions_[plugin_id]++;

    char* command = NULL;
    {
        int len_without_null = snprintf(
            NULL, 0, "%s %s.%i",
            plugin_info->compile_command, plugin_info->shared_object_path, lib_versions_[plugin_id]
        );
        alwaysAssert(len_without_null > 0);

        int buf_size = len_without_null + 1;

        command = (char*)mallocAsserted((size_t)len_without_null);
        len_without_null = snprintf(
            command, (size_t)buf_size, "%s %s.%i",
            plugin_info->compile_command, plugin_info->shared_object_path, lib_versions_[plugin_id]
        );
        alwaysAssert(len_without_null < buf_size);
    }
    defer(free(command));

    LOG_F(INFO, "Compiling plugin with command `%s`.", command);

    errno = 0;
    int ret = system(command);

    if (ret != 0) {
        int err = errno;
        const char* err_description = strerror(err);
        if (err_description == NULL) err_description = "(NO ERROR DESCRIPTION PROVIDED)";
        LOG_F(
            ERROR, "Failed to compile shared library `%s`; return code %i, errno %i, strerror(): `%s`.",
            plugin_info->shared_object_path, ret, err, err_description
        );

        return false;
    }

    bool success = loadLib(plugin_id);
    return success;
}

//
// ===========================================================================================================
//

} // namespace
