#ifndef _SORT_HPP
#define _SORT_HPP

// #include "types.hpp"
// #include "math_util.hpp"
// #include "thread_pool.hpp"

//
// ===========================================================================================================
//

struct KeyVal {
    u32 key;
    u32 val;
};

void mergeSort(
    const u32fast arr_size,
    KeyVal *const p_arr,
    KeyVal *const p_scratch,
    const u32fast skip_to_bucket_size = 1
);

void mergeSortMultiThreaded(
    thread_pool::ThreadPool* thread_pool,
    const u32fast thread_count,
    const u32fast arr_size,
    KeyVal *const p_keys,
    KeyVal *const p_scratch
);

//
// ===========================================================================================================
//

#endif // include guard
