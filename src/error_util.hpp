#ifndef _ERROR_UTIL_HPP
#define _ERROR_UTIL_HPP

//
// ===========================================================================================================
//

void _alwaysAssert(bool condition, const char* file, int line);
#define alwaysAssert(condition) _alwaysAssert(condition, __FILE__, __LINE__)

void _assertErrno(bool condition, const char* file, int line);
#define assertErrno(condition) _assertErrno(condition, __FILE__, __LINE__)

//
// ===========================================================================================================
//

#endif // include guard
