#ifndef _ALLOC_UTIL_HPP
#define _ALLOC_UTIL_HPP

//
// ===========================================================================================================
//

// #include <cstdlib>
// #include <cassert>
// #include <loguru/loguru.hpp>
// #include "error_util.hpp"
// #include "types.hpp"
// #include "math_util.hpp"


inline void* mallocAsserted(size_t size) {

    assert(size != 0);

    void* ptr = malloc(size);
    assertErrno(ptr != NULL);

    return ptr;
}


inline void* reallocAsserted(void* old_ptr, size_t size) {

    assert(size != 0);

    void* new_ptr = realloc(old_ptr, size);
    assertErrno(new_ptr != NULL);

    return new_ptr;
}


#define mallocArray(COUNT, TYPE) (TYPE*)mallocAsserted(COUNT * sizeof(TYPE))
#define reallocArray(PTR, COUNT, TYPE) (TYPE*)reallocAsserted(PTR, COUNT * sizeof(TYPE))


template<typename T>
struct ArrayList {
    T* ptr;
    u32 size;
    u32 capacity;

    static ArrayList<T> create() {
        return ArrayList { .ptr = NULL, .size = 0, .capacity = 0 };
    }

    static ArrayList<T> withCapacity(u32 cap) {
        T* ptr = mallocArray(cap, T);
        return ArrayList { .ptr = ptr, .size = 0, .capacity = cap };
    }

    static ArrayList<T> withCapacityAndSize(u32 size) {
        ArrayList<T> list = ArrayList<T>::withCapacity(size);
        list.size = size;
        return list;
    }

    void free() {

        T* p = this->ptr;
        if (p == NULL) return;

        std::free(p);
        this->ptr = NULL;
        this->size = 0;
        this->capacity = 0;
    }

    void reserve(u32 count) {
        if (count <= this->capacity) return;
        this->ptr = reallocArray(this->ptr, count, T);
        this->capacity = count;
    }

    /// Reserve space for pushing `count` more elements.
    /// This reserves `this->size + count`, NOT `this->capacity + count`.
    void reserveAdditional(u32 count) {
        if (count == 0) return;
        this->reserve(this->size + count);
    }

    T* pushUninitialized(void) {

        const u32 old_size = this->size;

        if (old_size == this->capacity) this->reserveAdditional(math::max(1, old_size / 2));

        this->size++;
        return this->ptr + old_size;
    }

    void push(const T& val) {

        const u32 sz = this->size;

        if (sz == this->capacity) this->reserveAdditional(math::max(1, sz / 2));

        this->ptr[sz] = val;
        this->size++;
    }

    void resetSize() {
        this->size = 0;
    }
};

//
// ===========================================================================================================
//

#endif // include guard
