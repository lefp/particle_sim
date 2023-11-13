#include <cstdio>
#include <cstdlib>

#include "types.hpp"
#include "log_stub.hpp"

void abortWithMessage(const char* msg) {
    logging::error("Aborting. Message: `%s`", msg);
    abort();
}

void _alwaysAssert(bool condition, const char* file, u64 line) {

    if (condition) return;

    logging::error("Assertion failed! File `%s`, line %lu\n", file, line);
    abort();
}

