#ifndef _ALLOC_UTIL_HPP
#define _ALLOC_UTIL_HPP

// #include <cstdlib>
// #include <cassert>
// #include <loguru/loguru.hpp>
// #include "error_util.hpp"
// #include "types.hpp"
// #include "math_util.hpp"

//
// ===========================================================================================================
//

// TODO FIXME these all should probably be static inline, not just inline

static inline size_t roundUpMultiple(size_t number_to_round, size_t multiple_of) {
    assert(multiple_of != 0);
    // TODO FIXME verify that this is correct
    return ((number_to_round + multiple_of - 1) / multiple_of) * multiple_of;
};


inline void* mallocAsserted(size_t size) {

    assert(size != 0);

    void* ptr = malloc(size);
    assertErrno(ptr != NULL);

    return ptr;
}


inline void* callocAsserted(size_t elem_count, size_t elem_size) {

    assert(elem_size != 0);
    assert(elem_count != 0);

    void* ptr = calloc(elem_count, elem_size);
    assertErrno(ptr != NULL);

    return ptr;
}


inline void* reallocAsserted(void* old_ptr, size_t size) {

    assert(size != 0);

    void* new_ptr = realloc(old_ptr, size);
    assertErrno(new_ptr != NULL);

    return new_ptr;
}


inline void* allocAlignedAsserted(size_t alignment, size_t size) {

    assert(size != 0);

    /// aligned_alloc: "The `size` parameter must be an integral multiple of `alignment`." (cppreference.com)
    size = roundUpMultiple(size, alignment);

    void* ptr = aligned_alloc(alignment, size);
    assertErrno(ptr != NULL);

    return ptr;
}


#define mallocArray(COUNT, TYPE) (TYPE*)mallocAsserted(COUNT * sizeof(TYPE))
#define callocArray(COUNT, TYPE) (TYPE*)callocAsserted(COUNT, sizeof(TYPE))
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

    T* pushZeroed(void) {

        const u32 old_size = this->size;

        if (old_size == this->capacity) this->reserveAdditional(math::max(1, old_size / 2));

        this->size++;
        T* new_ptr = this->ptr + old_size;

        memset(new_ptr, 0, sizeof(T));

        return new_ptr;
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

    void pop() {
        this->size--;
    };
};

//
// ===========================================================================================================
//

#endif // include guard
