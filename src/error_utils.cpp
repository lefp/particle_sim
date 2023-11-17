#include <loguru.hpp>

#include "types.hpp"

void _alwaysAssert(bool condition, const char* file, int line) {

    if (condition) return;

    ABORT_F("Assertion failed! File `%s`, line %i\n", file, line);
}

