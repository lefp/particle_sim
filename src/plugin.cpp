#include <dlfcn.h>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <cinttypes>

#include <glm/glm.hpp>
#include <loguru/loguru.hpp>
#include <tracy/tracy/Tracy.hpp>

#include "types.hpp"
#include "defer.hpp"
#include "error_util.hpp"
#include "math_util.hpp"
#include "alloc_util.hpp"
#include "str_util.hpp"
#include "file_watch.hpp"
#include "plugin.hpp"

#include "VulkanMemoryAllocator/vk_mem_alloc.h"
#include "vk_procs.hpp"
#include "vulkan_context.hpp"
#include "../build/A_generatePluginHeaders/plugin_infos.hpp"

namespace plugin {

//
// ===========================================================================================================
//

struct DynamicLibrary {
    void* p_procs_struct;
    void* dl_handle;
};

/// The elements of each plugin's ArrayList are indexed by the versions of that plugin.
static ArrayList<DynamicLibrary> plugins_[PluginID_COUNT] {};
static filewatch::Watchlist plugin_watchlists_[PluginID_COUNT] {};
static bool initialized_ = false;

//
// ===========================================================================================================
//


static inline void* allocProcsStruct_Zeroed_Asserted(PluginID plugin_id) {

    const plugin_infos::PluginProcStructInfo* proc_struct_info =
        &plugin_infos::PLUGIN_PROC_STRUCT_INFOS[plugin_id];

    void* ptr =  allocAlignedAsserted(proc_struct_info->alignment, proc_struct_info->size);
    memset(ptr, 0, proc_struct_info->size);

    return ptr;
}


[[nodiscard]] static bool loadLib(PluginID plugin_id, u32fast version_number, DynamicLibrary* p_lib_out) {

    ZoneScoped;

    assert(initialized_);
    alwaysAssert(0 <= plugin_id and plugin_id < PluginID_COUNT);


    const plugin_infos::PluginReloadInfo* plugin_info = &plugin_infos::PLUGIN_RELOAD_INFOS[plugin_id];


    char* lib_path = allocSprintf("%s.%" PRIuFAST32, plugin_info->shared_object_path, version_number);
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


    void* p_procs_struct = allocProcsStruct_Zeroed_Asserted(plugin_id);

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


    *p_lib_out = DynamicLibrary {
        .p_procs_struct = p_procs_struct,
        .dl_handle = dl_handle,
    };

    return true;
};

/// Returns a pointer to a struct of pointers.
/// The type of struct depends on the plugin you load.
///     E.g. fluid sim would be FluidSimProcs
extern const void* load(PluginID plugin_id) {

    ZoneScoped;

    assert(initialized_);
    alwaysAssert(0 <= plugin_id and plugin_id < PluginID_COUNT);

    if (plugins_[plugin_id].size != 0) {
        ABORT_F("load() called on plugin ID %i, but that plugin was already loaded.", plugin_id);
    }

    DynamicLibrary* p_new_lib = plugins_[plugin_id].pushZeroed();
    bool success = loadLib(plugin_id, 0, p_new_lib);
    if (!success) return NULL;

    return p_new_lib->p_procs_struct;
}

static bool runCommand(const char* command) {

    ZoneScoped;

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

extern const void* reload(PluginID plugin_id) {

    ZoneScoped;

    assert(initialized_);
    alwaysAssert(0 <= plugin_id and plugin_id < PluginID_COUNT);

    if (plugins_[plugin_id].size == 0) {
        ABORT_F("reload() called on plugin ID %i, but that plugin wasn't loaded.", plugin_id);
    }


    // Clear pending modification events, because we'll be up-to-date after this reload.
    {
        filewatch::Watchlist watchlist = plugin_watchlists_[plugin_id];
        if (watchlist != NULL) filewatch::clearEvents(watchlist);
    }


    u32fast new_version_number = plugins_[plugin_id].size;

    const plugin_infos::PluginReloadInfo* plugin_info = &plugin_infos::PLUGIN_RELOAD_INFOS[plugin_id];

    {
        ZoneScopedN("Compile plugin");

        char* command = allocSprintf("%s %s", plugin_info->compile_script, plugin_info->name);
        defer(free(command));

        LOG_F(INFO, "Compiling plugin with ID %i using command `%s`.", plugin_id, command);
        {
            // TODO Capture stdout+stderr so you can properly log the compiler errors in case of failure.
            bool success = runCommand(command);
            if (!success) {
                LOG_F(ERROR, "Failed to compile plugin with ID %i.", plugin_id);
                return NULL;
            }
        }
    }
    {
        ZoneScopedN("Link plugin");

        char* command = allocSprintf(
            "%s %s %" PRIuFAST32, plugin_info->link_script, plugin_info->name, new_version_number
        );
        defer(free(command));

        LOG_F(INFO, "Linking plugin with ID %i using command `%s`.", plugin_id, command);
        {
            // TODO Capture stdout+stderr so you can properly log the linker errors in case of failure.
            bool success = runCommand(command);
            if (!success) {
                LOG_F(ERROR, "Failed to link plugin with ID %i.", plugin_id);
                return NULL;
            }
        }
    }


    DynamicLibrary* p_new_lib = plugins_[plugin_id].pushZeroed();
    {
        bool success = loadLib(plugin_id, new_version_number, p_new_lib);
        if (!success) {
            plugins_[plugin_id].pop();
            return NULL;
        }
    }

    return p_new_lib->p_procs_struct;
}

extern void init(void) {
    for (u32fast i = 0; i < PluginID_COUNT; i++) {
        plugins_[i] = ArrayList<DynamicLibrary>::withCapacity(1);
    }
    initialized_ = true;
}

extern u32fast getLatestVersionNumber(PluginID plugin_id) {
    assert(initialized_);
    assert(0 <= plugin_id and plugin_id < PluginID_COUNT);

    return plugins_[plugin_id].size - 1;
};

extern void* getProcsVersioned(PluginID plugin_id, u32fast version) {
    assert(initialized_);
    assert(0 <= plugin_id and plugin_id < PluginID_COUNT);
    alwaysAssert(version < plugins_[plugin_id].size);

    return plugins_[plugin_id].ptr[version].p_procs_struct;
};

extern bool setFilewatchEnabled(PluginID plugin_id, bool enable) {

    alwaysAssert(0 <= plugin_id and plugin_id < PluginID_COUNT);

    filewatch::Watchlist current_watchlist = plugin_watchlists_[plugin_id];
    if ((current_watchlist != NULL) == enable) return true; // already set to the correct setting

    if (enable) {

        filewatch::Watchlist new_watchlist = filewatch::createWatchlist();
        if (new_watchlist == NULL) return false;

        const plugin_infos::PluginReloadInfo* plugin_info = &plugin_infos::PLUGIN_RELOAD_INFOS[plugin_id];
        for (u32fast i = 0; i < plugin_info->watch_filepath_count; i++) {
            filewatch::addFileToModificationWatchlist(new_watchlist, plugin_info->p_watch_filepaths[i]);
        }

        plugin_watchlists_[plugin_id] = new_watchlist;
    }
    else {
        filewatch::destroyWatchlist(current_watchlist);
        plugin_watchlists_[plugin_id] = NULL;
    }

    return true;
};

extern const void* reloadIfModified(PluginID plugin_id, bool* p_success_out) {

    assert(initialized_);
    assert(0 <= plugin_id and plugin_id < PluginID_COUNT);

    filewatch::Watchlist watchlist = plugin_watchlists_[plugin_id];
    CHECK_F(
        watchlist != NULL,
        "reloadIfModified() called on plugin ID %i, but source file watching is not enabled for that plugin.",
        plugin_id
    );

    u32 event_count = 0;
    const filewatch::FileID* _p_events;
    filewatch::poll(watchlist, &event_count, &_p_events);

    if (event_count == 0) {
        *p_success_out = true;
        return NULL;
    };

    const void* ptr = reload(plugin_id);
    *p_success_out = ptr != NULL;
    return ptr;
};

//
// ===========================================================================================================
//

} // namespace
