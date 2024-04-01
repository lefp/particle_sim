#ifndef _SORT_HPP
#define _SORT_HPP

// #include "types.hpp"
// #include "math_util.hpp"
// #include "thread_pool.hpp"

//
// ===========================================================================================================
//

void mergeSort(
    const u32fast arr_size,
    u32 *const p_keys,
    u32 *const p_vals,
    u32 *const p_scratch1,
    u32 *const p_scratch2,
    const u32fast skip_to_bucket_size = 1
);

void mergeSortMultiThreaded(
    thread_pool::ThreadPool* thread_pool,
    const u32fast thread_count,
    const u32fast arr_size,
    u32 *const p_keys,
    u32 *const p_vals,
    u32 *const p_scratch1,
    u32 *const p_scratch2
);

//
// ===========================================================================================================
//

#endif // include guard
