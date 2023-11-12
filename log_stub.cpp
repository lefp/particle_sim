#include <cstdio>
#include <cstdarg>

#include "log_stub.hpp"

namespace log {
    void info(const char* fmt, ...) {
        va_list fmt_args;
        va_start(fmt_args, fmt);

        fprintf(stderr, "\033[34;1minfo: \033[0m");
        vfprintf(stderr, fmt, fmt_args);
        fprintf(stderr, "\n");

        va_end(fmt_args);
    }
    void error(const char* fmt, ...) {
        va_list fmt_args;
        va_start(fmt_args, fmt);

        fprintf(stderr, "\033[31;1merror: \033[0m");
        vfprintf(stderr, fmt, fmt_args);
        fprintf(stderr, "\n");

        va_end(fmt_args);
    }
}
