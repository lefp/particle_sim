#ifndef _TYPES_HPP
#define _TYPES_HPP

//
// ===========================================================================================================
//

#include <cstdint>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t  i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float  f32;
typedef double f64;

// "the fastest uint with at least sizeof(u32)"; use when you don't care if it's 32-bit or 64-bit
typedef uint_fast32_t u32fast;

//
// ===========================================================================================================
//

#endif // include guard
