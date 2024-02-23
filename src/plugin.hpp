#ifndef _PLUGIN_HPP
#define _PLUGIN_HPP

#include "../build/A_generatePluginHeaders/plugin_ids.hpp"

namespace plugin {

//
// ===========================================================================================================
//

[[nodiscard]] const void* load(PluginID plugin_id);
[[nodiscard]] bool reload(PluginID plugin_id);

/// NOTE: You might need to bring the xxxProcs type into your namespace for this to work.
///     E.g.: `using fluid_sim::FluidSimProcs;`.
#define PLUGIN_LOAD(VARIABLE_TO_INITIALIZE, PLUGIN_NAME_TITLECASE) \
    {                                                                                                        \
        VARIABLE_TO_INITIALIZE =                                                                             \
            ((const PLUGIN_NAME_TITLECASE ## Procs*) plugin::load(PluginID_ ## PLUGIN_NAME_TITLECASE));      \
    }

//
// ===========================================================================================================
//

} // namespace

#endif // include guard
