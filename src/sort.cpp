#include <cstring>

#include <tracy/tracy/Tracy.hpp>

#include "types.hpp"
#include "math_util.hpp"
#include "error_util.hpp"
#include "thread_pool.hpp"
#include "sort.hpp"
#include "defer.hpp"

// TODO FIXME we should fuzz-test these algorithms

//
// ===========================================================================================================
//

#define SWAP(a, b) \
{ \
    typeof(a) _tmp = a; \
    a = b; \
    b = _tmp; \
}

static inline void mergeSort_merge(

    KeyVal* arr_in,
    KeyVal* arr_out,

    u32fast idx_a,
    u32fast idx_b,
    u32fast idx_dst,

    u32fast idx_a_end,
    u32fast idx_b_end,
    u32fast idx_dst_end
) {

    for (; idx_dst < idx_dst_end; idx_dst++)
    {
        u32 key_a;
        if (idx_a < idx_a_end) key_a = arr_in[idx_a].key;
        else key_a = UINT32_MAX;

        u32 key_b;
        if (idx_b < idx_b_end) key_b = arr_in[idx_b].key;
        else key_b = UINT32_MAX;

        if (key_a <= key_b)
        {
            arr_out[idx_dst] = arr_in[idx_a];
            idx_a++;
        }
        else
        {
            arr_out[idx_dst] = arr_in[idx_b];
            idx_b++;
        }
    }
}

extern void mergeSort(
    const u32fast arr_size,
    KeyVal *const p_arr,
    KeyVal *const p_scratch,
    const u32fast skip_to_bucket_size
) {

    ZoneScoped;

    if (arr_size < 2) return;


    KeyVal* arr1 = p_arr;
    KeyVal* arr2 = p_scratch;


    for (u32fast bucket_size = skip_to_bucket_size; bucket_size < arr_size; bucket_size *= 2)
    {
        const u32fast bucket_count = arr_size / bucket_size + (arr_size % bucket_size != 0);

        for (u32fast bucket_idx = 0; bucket_idx < bucket_count; bucket_idx += 2)
        {
            u32fast idx_a = (bucket_idx    ) * bucket_size;
            u32fast idx_b = (bucket_idx + 1) * bucket_size;
            u32fast idx_dst = idx_a;

            const u32fast idx_a_max = math::min(idx_a + bucket_size, arr_size);
            const u32fast idx_b_max = math::min(idx_b + bucket_size, arr_size);
            const u32fast idx_dst_max = math::min(idx_dst + 2*bucket_size, arr_size);

            mergeSort_merge(
                arr1,
                arr2,
                idx_a, idx_b, idx_dst,
                idx_a_max, idx_b_max, idx_dst_max
            );
        }

        SWAP(arr1, arr2);
    }

    if (arr1 != p_arr)
    {
        // TODO profile this
        memcpy(p_arr, arr1, arr_size * sizeof(KeyVal));
    }
}

struct MergeSortThreadParams {
    u32fast array_size;
    u32fast skip_to_bucket_size;
    KeyVal* p_array;
    KeyVal* p_scratch;
};
static void mergeSortMultiThreaded_thread(void* p_params_struct)
{
    const MergeSortThreadParams* params = (const MergeSortThreadParams*)p_params_struct;

    mergeSort(
        params->array_size,
        params->p_array,
        params->p_scratch,
        params->skip_to_bucket_size
    );
};

extern void mergeSortMultiThreaded(
    thread_pool::ThreadPool* thread_pool,
    u32fast thread_count,
    const u32fast arr_size,
    KeyVal *const p_arr,
    KeyVal *const p_scratch
) {

    ZoneScoped;

    assert(thread_count > 0);

    if (arr_size < 2) return;

    if (arr_size < thread_count or thread_count == 1)
    {
        mergeSort(arr_size, p_arr, p_scratch);
        return;
    }

    // TODO do some minimum block size thing, where each thread must get some minimum number `m` of elements
    // to sort; if there are too few elements, spawn fewer threads.
    // Because spawning a thread just to have it sort 2 values is dumb.


    // OPTIMIZE: Instead of heap-allocating here (this function may be called every frame!), we can write a
    // `mergeSortMultithreaded_init()` procedure that does the alloc once.

    thread_pool::TaskId* tasks = (thread_pool::TaskId*)calloc(thread_count, sizeof(thread_pool::TaskId));
    defer(free(tasks));

    MergeSortThreadParams* thread_params = (MergeSortThreadParams*)calloc(thread_count, sizeof(MergeSortThreadParams));
    defer(free(thread_params));


    u32fast block_size = arr_size / thread_count;
    {
        u32fast remaining_element_count = arr_size;

        {
            ZoneScopedN("spawn threads");

            KeyVal* param_p_arr = p_arr;
            KeyVal* param_p_scratch = p_scratch;

            for (u32fast i = 0; i < thread_count; i++)
            {
                thread_params[i] = MergeSortThreadParams {
                    .array_size = block_size,
                    .skip_to_bucket_size = 1,
                    .p_array = param_p_arr,
                    .p_scratch = param_p_scratch,
                };

                tasks[i] = thread_pool::enqueueTask(thread_pool, mergeSortMultiThreaded_thread, &thread_params[i]);

                param_p_arr += block_size;
                param_p_scratch += block_size;

                assert(remaining_element_count >= block_size);
                remaining_element_count -= block_size;
            }
        }

        if (remaining_element_count > 0)
        {
            const u32fast idx_start = arr_size - remaining_element_count;
            mergeSort(
                remaining_element_count,
                p_arr + idx_start,
                p_scratch + idx_start
            );
        }

        {
            ZoneScopedN("join threads");

            for (u32 i = 0; i < thread_count; i++)
            {
                thread_pool::waitForTask(thread_pool, tasks[i]);
            }
        }
    }

    while (true)
    {
        u32fast remaining_block_size = arr_size;

        u32fast old_block_size = block_size;
        block_size *= 2;
        thread_count = arr_size / block_size;
        if (thread_count < 2) break;

        u32fast idx_start = 0;

        {
            ZoneScopedN("spawn threads");

            for (u32fast i = 0; i < thread_count; i++)
            {
                thread_params[i] = MergeSortThreadParams {
                    .array_size = block_size,
                    .skip_to_bucket_size = old_block_size,
                    .p_array = p_arr + idx_start,
                    .p_scratch = p_scratch + idx_start,
                };

                tasks[i] = thread_pool::enqueueTask(thread_pool, mergeSortMultiThreaded_thread, &thread_params[i]);

                idx_start += block_size;
                assert(remaining_block_size >= block_size);
                remaining_block_size -= block_size;
            }

            assert(idx_start + remaining_block_size == arr_size);
        }

        if (remaining_block_size > 0)
        {
            mergeSort(
                remaining_block_size,
                p_arr + idx_start,
                p_scratch + idx_start,
                old_block_size
            );
        }

        {
            ZoneScopedN("join threads");
            for (u32fast i = 0; i < thread_count; i++) thread_pool::waitForTask(thread_pool, tasks[i]);
        }
    }

    assert(block_size <= arr_size);
    mergeSort(arr_size, p_arr, p_scratch, block_size / 2);
};
