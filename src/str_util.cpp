#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <cassert>

#include "types.hpp"
#include "math_util.hpp"
#include "error_util.hpp"
#include "alloc_util.hpp"
#include "str_util.hpp"

extern char* allocSprintf(const char *__restrict format, ...) {

    int len_without_null_terminator = 0;
    {
        va_list args;
        va_start(args, format);
        len_without_null_terminator = vsnprintf(NULL, 0, format, args);
        va_end(args);
    }
    assert(len_without_null_terminator != 0);
    alwaysAssert(len_without_null_terminator > 0);

    size_t buffer_size = (size_t)(len_without_null_terminator + 1);
    char* buffer = mallocArray(buffer_size, char);

    {
        va_list args;
        va_start(args, format);
        len_without_null_terminator = vsnprintf(buffer, buffer_size, format, args);
        va_end(args);
    }
    assert(len_without_null_terminator != 0);
    alwaysAssert(len_without_null_terminator > 0);
    assert(len_without_null_terminator < (int)buffer_size);

    return buffer;
}

