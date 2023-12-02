#include <cerrno>
#include <cstring>

#include <loguru/loguru.hpp>

#include "types.hpp"
#include "error_util.hpp"

void _alwaysAssert(bool condition, const char* file, int line) {

    if (condition) return;

    ABORT_F("Assertion failed! File `%s`, line %i\n", file, line);
}

void _assertErrno(bool condition, const char* file, int line) {

    if (condition) return;

    const char* err_description = strerror(errno);
    alwaysAssert(err_description != NULL);

    ABORT_F(
        "Assertion failed! File `%s`, line %i, errno %i, strerror `%s`.",
        file, line, errno, err_description
    );
};
