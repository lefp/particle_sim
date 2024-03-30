#include <cstring>

#include <tracy/tracy/Tracy.hpp>

#include "types.hpp"
#include "math_util.hpp"
#include "sort.hpp"

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

    u32* keys_1,
    u32* keys_2,
    u32* vals_1,
    u32* vals_2,

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
        if (idx_a < idx_a_end) key_a = keys_1[idx_a];
        else key_a = UINT32_MAX;

        u32 key_b;
        if (idx_b < idx_b_end) key_b = keys_1[idx_b];
        else key_b = UINT32_MAX;

        if (key_a <= key_b)
        {
            keys_2[idx_dst] = keys_1[idx_a];
            vals_2[idx_dst] = vals_1[idx_a];
            idx_a++;
        }
        else
        {
            keys_2[idx_dst] = keys_1[idx_b];
            vals_2[idx_dst] = vals_1[idx_b];
            idx_b++;
        }
    }
}

extern void mergeSort(
    const u32fast arr_size,
    u32 *const p_keys,
    u32 *const p_vals,
    u32 *const p_scratch1,
    u32 *const p_scratch2,
    const u32 skip_to_bucket_size
) {

    ZoneScoped;

    if (arr_size < 2) return;


    u32* keys_arr1 = p_keys;
    u32* keys_arr2 = p_scratch1;

    u32* vals_arr1 = p_vals;
    u32* vals_arr2 = p_scratch2;


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
                keys_arr1,
                keys_arr2,
                vals_arr1,
                vals_arr2,
                idx_a, idx_b, idx_dst,
                idx_a_max, idx_b_max, idx_dst_max
            );
        }

        SWAP(keys_arr1, keys_arr2);
        SWAP(vals_arr1, vals_arr2);
    }

    if (keys_arr1 != p_keys)
    {
        memcpy(p_keys, keys_arr1, arr_size * sizeof(u32));
    }
    if (vals_arr1 != p_vals)
    {
        memcpy(p_vals, vals_arr1, arr_size * sizeof(u32));
    }
}
