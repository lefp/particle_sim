// #include <cstdlib>
// #include <assert.h>

template<typename T>
inline T* mallocArray(size_t count) {

    assert(count != 0);

    T* ptr = (T*)malloc(count * sizeof(T));
    assertErrno(ptr != NULL);

    return ptr;
}
