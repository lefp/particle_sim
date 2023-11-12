#include "types.hpp"

extern void _alwaysAssert(bool condition, const char* file, u64 line);
#define alwaysAssert(condition) _alwaysAssert(condition, __FILE__, __LINE__)



