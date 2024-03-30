#ifndef _MATH_UTIL_HPP
#define _MATH_UTIL_HPP

// #include "types.hpp"

//
// ===========================================================================================================
//

namespace math {
    template <typename T>
    static inline T min(T a, T b) {
        return a < b ? a : b;
    }

    template <typename T>
    static inline T max(T a, T b) {
        return a > b ? a : b;
    }

    template <typename T>
    static inline T clamp(T val, T min, T max) {
        return math::min(max, math::max(min, val));
    }
}

//
// ===========================================================================================================
//

#endif // include guard
