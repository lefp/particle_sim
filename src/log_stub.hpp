//! Temporary logger implementation; TODO come up with a better one or find a good third-party library.

#ifndef _LOG_STUB_HPP
#define _LOG_STUB_HPP

namespace logging {
    void info(const char* fmt, ...);
    void error(const char* fmt, ...);
};

#endif // include guard
