#ifndef _PLUGIN_HPP
#define _PLUGIN_HPP

// #include "types.hpp"
#include "../build/A_generatePluginHeaders/plugin_ids.hpp"

namespace plugin {

//
// ===========================================================================================================
//

void init(void);
[[nodiscard]] const void* load(PluginID plugin_id);

/// Returns a pointer to the new procedures struct.
/// Old pointers obtained from load/reload are still valid; they're just for older versions.
[[nodiscard]] const void* reload(PluginID plugin_id);

/// You must have successfully enabled file watching for a plugin before using this.
/// If the plugin was not modified, returns NULL and sets success=true.
/// If the reload fails, returns NULL and sets success=false.
/// If the reload succeeds, returns a valid pointer and sets success=true.
[[nodiscard]] const void* reloadIfModified(PluginID plugin_id, bool* success);

[[nodiscard]] u32fast getLatestVersionNumber(PluginID plugin_id);
[[nodiscard]] void* getProcsVersioned(PluginID plugin_id, u32fast version);

/// Returns true iff successful.
[[nodiscard]] bool setFilewatchEnabled(PluginID plugin_id, bool enable);

/// NOTE: You might need to bring the xxxProcs type into your namespace for this to work.
///     E.g.: `using fluid_sim::FluidSimProcs;`.
#define PLUGIN_LOAD(VARIABLE_TO_INITIALIZE, PLUGIN_NAME_TITLECASE) \
    {                                                                                                        \
        VARIABLE_TO_INITIALIZE =                                                                             \
            ((const PLUGIN_NAME_TITLECASE ## Procs*) plugin::load(PluginID_ ## PLUGIN_NAME_TITLECASE));      \
    }

#define PLUGIN_RELOAD(VARIABLE_TO_INITIALIZE, PLUGIN_NAME_TITLECASE) \
    {                                                                                                        \
        VARIABLE_TO_INITIALIZE =                                                                             \
            ((const PLUGIN_NAME_TITLECASE ## Procs*) plugin::reload(PluginID_ ## PLUGIN_NAME_TITLECASE));    \
    }

#define PLUGIN_RELOAD_IF_MODIFIED(VARIABLE_TO_INITIALIZE, PLUGIN_NAME_TITLECASE, SUCCESS_PTR_VARIABLE) \
    {                                                                                                        \
        VARIABLE_TO_INITIALIZE = (                                                                           \
            (const PLUGIN_NAME_TITLECASE ## Procs*)                                                          \
            plugin::reloadIfModified(PluginID_ ## PLUGIN_NAME_TITLECASE, SUCCESS_PTR_VARIABLE)               \
        );                                                                                                   \
    }

//
// ===========================================================================================================
//

} // namespace

#endif // include guard
