#include <dlfcn.h>

#include <shaderc/shaderc.h>
#include <loguru/loguru.hpp>

#include "libshaderc_procs.hpp"
#include "error_util.hpp"

//
// ===========================================================================================================
//

ShadercProcs libshaderc_procs_ {};

//
// ===========================================================================================================
//

bool ShadercProcs::init(void) {

    // OPTIMIZE: decide whether we want RTLD_NOW or RTLD_LAZY
    void* shaderc_dl_handle = dlopen("libshaderc_shared.so", RTLD_NOW);

    if (shaderc_dl_handle == NULL) {

        const char* err_description = dlerror();
        if (err_description == NULL) err_description = "(NO ERROR DESCRIPTION PROVIDED)";

        LOG_F(ERROR, "Failed to load libshaderc. dlerror(): `%s`.", err_description);

        return false;
    }


    #define INITIALIZE_PROC_PTR(PROC_NAME) \
        {                                                                                                    \
            const char* proc_name = "shaderc_" #PROC_NAME;                                                   \
            void* proc_ptr = dlsym(shaderc_dl_handle, proc_name);                                            \
            if (proc_ptr == NULL) {                                                                          \
                                                                                                             \
                const char* err_description = dlerror();                                                     \
                if (err_description == NULL) err_description = "(NO ERROR DESCRIPTION PROVIDED)";            \
                                                                                                             \
                LOG_F(ERROR, "Failed to load proc `%s`. dlerror(): `%s`.", proc_name, err_description);      \
                                                                                                             \
                int result = dlclose(shaderc_dl_handle);                                                     \
                alwaysAssert(result == 0);                                                                   \
                                                                                                             \
                return false;                                                                                \
            }                                                                                                \
            this->PROC_NAME = (typeof(this->PROC_NAME))proc_ptr;                                             \
        }

    FOR_EACH_SHADERC_PROC(INITIALIZE_PROC_PTR);

    #undef INITIALIZE_PROC_PTR


    return true;
}
