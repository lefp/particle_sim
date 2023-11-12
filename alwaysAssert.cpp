#include <cstdlib>
#include <cstdio>

#include "log_stub.hpp"
#include "alwaysAssert.hpp"

extern void _alwaysAssert(bool condition, const char* file, u64 line) {

    if (condition) return;

    log::error("Assertion failed! File `%s`, line %lu\n", file, line);
    abort();
}

