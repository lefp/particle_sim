#ifndef _FILE_UTIL_HPP
#define _FILE_UTIL_HPP

namespace file_util {

//
// ===========================================================================================================
//

/// You own the returned buffer. You may free it using `free()`.
/// On error, either aborts or returns `NULL`.
[[nodiscard]] void* readEntireFile(const char* fname, size_t* size_out);

//
// ===========================================================================================================
//

} // namespace

#endif // include guard
