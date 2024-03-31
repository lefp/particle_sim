#include <cassert>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

#include <tracy/tracy/Tracy.hpp>

#include "types.hpp"
#include "error_util.hpp"
#include "thread_pool.hpp"

namespace thread_pool
{

//
// ===========================================================================================================
//

constexpr u32 IDX_NONE = UINT32_MAX;

struct Task
{
    // A task has 3 states, which it cycles through in this order:
    // 1. Free (not a task)
    // 2. Enqueued (waiting for a thread to begin executing it)
    // 3. InProgress (being executed by a thread)
    u32 freelist_next_idx; // valid if this task is Free; otherwise IDX_NONE
    u32 tasklist_next_idx; // valid if this task is Enqueued; otherwise IDX_NONE
    // If both of the above are IDX_NONE, the task is InProgress.

    PFN_TaskProc p_procedure;
    void* p_arg;

    u32 generation; // when a task completes, its generation is incremented
    pthread_cond_t finished_condition;
};

struct ThreadPool
{
    u32 thread_count;
    pthread_t* p_threads;

    Task* p_tasks;

    u32 tasklist_first_idx;
    u32 freelist_first_idx;

    // There are two conditions for signalling this variable:
    // 1. A new task has been enqueued.
    // 2. It is time for the threads to quit.
    //     In this case, `all_threads_should_quit` will be set to true before signalling.
    pthread_cond_t new_task_available_condition;
    bool all_threads_should_quit;

    pthread_mutex_t mutex;
};

static void* threadProcedure(void *const p_task_queue)
{
    ZoneScoped;

    ThreadPool *const queue = (ThreadPool*)p_task_queue;


    int result = pthread_mutex_lock(&queue->mutex);
    alwaysAssert(result == 0);

    while (true)
    {
        // wait for a task to be enqueued --------------------------------------------------------------------

        {
            ZoneScopedN("wait for task");

            while (queue->tasklist_first_idx == IDX_NONE)
            {
                result = pthread_cond_wait(&queue->new_task_available_condition, &queue->mutex);
                alwaysAssert(result == 0);

                if (queue->all_threads_should_quit)
                {
                    result = pthread_mutex_unlock(&queue->mutex);
                    alwaysAssert(result == 0);
                    pthread_exit(NULL);
                }
            }
        }

        // pop the task from the queue -----------------------------------------------------------------------

        u32 task_idx = queue->tasklist_first_idx;
        assert(task_idx != IDX_NONE);

        Task* task = &queue->p_tasks[task_idx];
        queue->tasklist_first_idx = task->tasklist_next_idx;
        task->tasklist_next_idx = IDX_NONE;

        result = pthread_mutex_unlock(&queue->mutex);
        alwaysAssert(result == 0);

        // execute the task ----------------------------------------------------------------------------------

        {
            ZoneScopedN("execute task");
            task->p_procedure(task->p_arg);
        }

        // mark the task complete ----------------------------------------------------------------------------

        result = pthread_mutex_lock(&queue->mutex);
        alwaysAssert(result == 0);

        task->generation++;
        task->freelist_next_idx = queue->freelist_first_idx;
        queue->freelist_first_idx = task_idx;

        result = pthread_cond_broadcast(&task->finished_condition);
        alwaysAssert(result == 0);
    }
}

extern ThreadPool* create(u32 thread_count, u32 max_queue_size)
{
    ZoneScoped;

    alwaysAssert(thread_count > 0);
    alwaysAssert(max_queue_size > 0);


    // We heap-allocate the queue (instead of just returning it on the stack) to ensure that any pointers to
    // it remain valid. This is because each thread in the pool will contain a pointer to the queue.
    ThreadPool* p_queue = (ThreadPool*)calloc(1, sizeof(ThreadPool));
    assertErrno(p_queue != NULL);


    int result = pthread_mutex_init(&p_queue->mutex, NULL);
    alwaysAssert(result == 0);

    result = pthread_mutex_lock(&p_queue->mutex);
    alwaysAssert(result == 0);


    result = pthread_cond_init(&p_queue->new_task_available_condition, NULL);
    alwaysAssert(result == 0);

    p_queue->tasklist_first_idx = IDX_NONE;
    p_queue->freelist_first_idx = 0;


    p_queue->p_tasks = (Task*)calloc(max_queue_size, sizeof(Task));
    assertErrno(p_queue->p_tasks != NULL);

    for (u32 i = 0; i < max_queue_size; i++)
    {
        Task* task = &p_queue->p_tasks[i];

        task->freelist_next_idx = i + 1;
        task->tasklist_next_idx = IDX_NONE;

        result = pthread_cond_init(&task->finished_condition, NULL);
        alwaysAssert(result == 0);
    }
    p_queue->p_tasks[max_queue_size - 1].freelist_next_idx = IDX_NONE;


    p_queue->all_threads_should_quit = false;

    p_queue->thread_count = thread_count;
    p_queue->p_threads = (pthread_t*)calloc(thread_count, sizeof(pthread_t));
    assertErrno(p_queue->p_threads != NULL);

    for (u32 i = 0; i < thread_count; i++)
    {
        result = pthread_create(&p_queue->p_threads[i], NULL, threadProcedure, p_queue);
        alwaysAssert(result == 0);
    }


    result = pthread_mutex_unlock(&p_queue->mutex);
    alwaysAssert(result == 0);

    return p_queue;
}

extern TaskId enqueueTask(ThreadPool* queue, PFN_TaskProc p_procedure, void* p_arg)
{
    ZoneScoped;


    int result = pthread_mutex_lock(&queue->mutex);
    alwaysAssert(result == 0);

    u32 task_idx = queue->freelist_first_idx;
    alwaysAssert(task_idx != IDX_NONE); // assert queue is not full
    Task* task = &queue->p_tasks[task_idx];

    queue->freelist_first_idx = task->freelist_next_idx;
    task->freelist_next_idx = IDX_NONE;

    task->tasklist_next_idx = queue->tasklist_first_idx;
    queue->tasklist_first_idx = task_idx;

    task->p_procedure = p_procedure;
    task->p_arg = p_arg;

    TaskId task_id {
        .idx = task_idx,
        .generation = task->generation
    };

    result = pthread_cond_signal(&queue->new_task_available_condition);
    alwaysAssert(result == 0);

    result = pthread_mutex_unlock(&queue->mutex);
    alwaysAssert(result == 0);

    return task_id;
}

extern void waitForTask(ThreadPool* queue, const TaskId task_id)
{
    ZoneScoped;


    int result = pthread_mutex_lock(&queue->mutex);
    alwaysAssert(result == 0);

    Task* p_task = &queue->p_tasks[task_id.idx];
    while (p_task->generation == task_id.generation)
    {
        result = pthread_cond_wait(&p_task->finished_condition, &queue->mutex);
        alwaysAssert(result == 0);
    }

    result = pthread_mutex_unlock(&queue->mutex);
    alwaysAssert(result == 0);
}

extern void destroy(ThreadPool* queue)
{
    ZoneScoped;


    int result = pthread_mutex_lock(&queue->mutex);
    alwaysAssert(result == 0);

    // wait for pending tasks to complete --------------------------------------------------------------------

    while (queue->tasklist_first_idx != IDX_NONE)
    {
        Task* task = &queue->p_tasks[queue->tasklist_first_idx];
        result = pthread_cond_wait(&task->finished_condition, &queue->mutex);
        alwaysAssert(result == 0);
    }

    // quit the threads --------------------------------------------------------------------------------------

    queue->all_threads_should_quit = true;
    result = pthread_cond_broadcast(&queue->new_task_available_condition);
    alwaysAssert(result == 0);

    result = pthread_mutex_unlock(&queue->mutex);
    alwaysAssert(result == 0);

    for (u32fast i = 0; i < queue->thread_count; i++)
    {
        result = pthread_join(queue->p_threads[i], NULL);
        alwaysAssert(result == 0);
    }

    // destroy everything ------------------------------------------------------------------------------------

    {
        u32 task_idx = queue->freelist_first_idx;
        assert(task_idx != IDX_NONE);

        while (task_idx != IDX_NONE)
        {
            result = pthread_cond_destroy(&queue->p_tasks[task_idx].finished_condition);
            alwaysAssert(result == 0);

            task_idx = queue->p_tasks[task_idx].tasklist_next_idx;
        }
    }

    free(queue->p_tasks);
    free(queue->p_threads);

    result = pthread_cond_destroy(&queue->new_task_available_condition);
    alwaysAssert(result == 0);

    result = pthread_mutex_destroy(&queue->mutex);
    alwaysAssert(result == 0);

    memset(queue, 0, sizeof(*queue));
}

//
// ===========================================================================================================
//

} // namespace
