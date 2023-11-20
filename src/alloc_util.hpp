#ifndef _ALLOC_UTIL_HPP
#define _ALLOC_UTIL_HPP

//
// ===========================================================================================================
//

// #include <cstdlib>
// #include <assert.h>

template<typename T>
inline T* mallocArray(size_t count) {

    assert(count != 0);

    T* ptr = (T*)malloc(count * sizeof(T));
    assertErrno(ptr != NULL);

    return ptr;
}

//
// ===========================================================================================================
//

#endif // include guard
