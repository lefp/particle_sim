#ifndef _MATH_UTIL_HPP
#define _MATH_UTIL_HPP

// #include "types.hpp"

//
// ===========================================================================================================
//

namespace math {
    inline u32 min(u32 a, u32 b) {
        return a < b ? a : b;
    }

    inline u32 max(u32 a, u32 b) {
        return a > b ? a : b;
    }

    inline u32 clamp(u32 val, u32 min, u32 max) {
        return math::min(max, math::max(min, val));
    }
}

//
// ===========================================================================================================
//

#endif // include guard
