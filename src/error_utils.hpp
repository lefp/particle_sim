void abortWithMessage(const char* msg);

void _alwaysAssert(bool condition, const char* file, u64 line);
#define alwaysAssert(condition) _alwaysAssert(condition, __FILE__, __LINE__)



