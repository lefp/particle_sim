// #include <cstdlib>
// #include <assert.h>

template<typename T>
inline T* mallocArray(size_t count) {

    assert(count != 0);

    T* ptr = (T*)malloc(count * sizeof(T));
    // TODO(peterlef): if the assertion fails, all we'll know is that `mallocArray` failed. It won't tell
    // us *where* it failed. We should set up a handler to catch the failed assertion and log a backtrace
    // or something.
    alwaysAssert(ptr != NULL);

    return ptr;
}
