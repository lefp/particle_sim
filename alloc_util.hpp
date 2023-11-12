// #include <cstdlib>
// #include <assert.h>

// TODO __attribute__((always_inline))?
template<typename T>
inline T* mallocArray(size_t count) {

    assert(count != 0);

    void* ptr = malloc(count * sizeof(T));
    alwaysAssert(ptr != NULL);

    return ptr;
}
