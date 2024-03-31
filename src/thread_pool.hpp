#ifndef _THREAD_POOL_HPP
#define _THREAD_POOL_HPP

#include "types.hpp"

namespace thread_pool {

//
// ===========================================================================================================
//

using TaskProc = void (void* p_arg);
using PFN_TaskProc = TaskProc*;

struct TaskId
{
    u32 idx;
    u32 generation;
};

struct ThreadPool;

ThreadPool* create(u32 thread_count, u32 max_queue_size);
void destroy(ThreadPool*);

TaskId enqueueTask(ThreadPool*, PFN_TaskProc p_procedure, void* p_arg);
void waitForTask(ThreadPool*, TaskId);

//
// ===========================================================================================================
//

} // namespace

#endif // include guard
