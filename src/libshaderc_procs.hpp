#ifndef _LIBSHADEC_PROCS_HPP
#define _LIBSHADEC_PROCS_HPP

// #include <shaderc/shaderc.h>

//
// ===========================================================================================================
//

#define FOR_EACH_SHADERC_PROC(X) \
    X(compile_into_spv) \
    X(compiler_initialize) \
    X(result_get_bytes) \
    X(result_get_compilation_status) \
    X(result_get_error_message) \
    X(result_get_length) \
    X(result_get_num_errors) \
    X(result_release)

//
// ===========================================================================================================
//

#define DECLARE_PROC_PTR(PROC_NAME) typeof(shaderc_##PROC_NAME)* PROC_NAME;

struct ShadercProcs {
    FOR_EACH_SHADERC_PROC(DECLARE_PROC_PTR)

    /// Returns whether it succeeded.
    [[nodiscard]] bool init(void);
};

#undef DECLARE_PROC_PTR

//
// ===========================================================================================================
//

extern ShadercProcs libshaderc_procs_;

//
// ===========================================================================================================
//

#endif // include guard
