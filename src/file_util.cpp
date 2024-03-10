#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <cstring>

#include <loguru/loguru.hpp>

#include "error_util.hpp"
#include "file_util.hpp"

namespace file_util {

//
// ===========================================================================================================
//

extern void* readEntireFile(const char* fname, size_t* size_out) {
    // OPTIMIZE: Maybe using `open()`, `fstat()`, and `read()` would be faster; because we don't need buffered
    // input, and maybe using `fseek()` to get the file size is unnecessarily slow.

    FILE* file = fopen(fname, "r");
    if (file == NULL) {
        LOG_F(ERROR, "Failed to open file `%s`; errno: `%i`, description: `%s`.", fname, errno, strerror(errno));
        return NULL;
    }

    int result = fseek(file, 0, SEEK_END);
    assertErrno(result == 0);

    size_t file_size;
    {
        long size = ftell(file);
        assertErrno(size >= 0);
        file_size = (size_t)size;
    }

    result = fseek(file, 0, SEEK_SET);
    assertErrno(result == 0);


    void* buffer = malloc(file_size);
    assertErrno(buffer != NULL);

    size_t n_items_read = fread(buffer, file_size, 1, file);
    alwaysAssert(n_items_read == 1);

    result = fclose(file);
    assertErrno(result == 0);


    *size_out = file_size;
    return buffer;
}

//
// ===========================================================================================================
//

} // namespace

