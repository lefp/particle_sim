#ifndef _DEFER_HPP
#define _DEFER_HPP

//
// ===========================================================================================================
//

// source: https://www.gingerbill.org/article/2015/08/19/defer-in-cpp

template <typename F>
struct _DeferStruct {
    F f;

    _DeferStruct(F ff) : f(ff) {}
    ~_DeferStruct() { f(); }
};

template <typename F>
static _DeferStruct<F> _defer_func(F f) {
    return _DeferStruct<F>(f);
}

#define DEFER_1(x, y) x##y
#define DEFER_2(x, y) DEFER_1(x, y)
#define DEFER_3(x)    DEFER_2(x, __COUNTER__)
#define defer(code)   auto DEFER_3(_defer_) = _defer_func([&](){code;})


// TODO: are destructors guaranteed to be called in reverse order of object declaration?

//
// ===========================================================================================================
//

#endif // include guard
