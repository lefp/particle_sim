#ifdef linux
    #include <sys/inotify.h>
    #include <unistd.h>
    #include <linux/limits.h>
#else
    #error("Whatever OS you're compiling for is not supported.")
#endif

#include <cassert>
#include <cerrno>
#include <cstring>

#include <loguru/loguru.hpp>

#include "types.hpp"
#include "file_watch.hpp"
#include "error_util.hpp"
#include "alloc_util.hpp"

namespace filewatch {

//
// ===========================================================================================================
//

struct WatchlistImpl {
    ArrayList<FileID> modified_files;
    int inotify_fd;
};

//
// ===========================================================================================================
//

Watchlist createWatchlist(void) {

    int inotify_fd = inotify_init1(IN_NONBLOCK);
    if (inotify_fd < 0) {
        LOG_F(ERROR, "Failed to initialize inotify. errno %d, strerror: `%s`.", errno, strerror(errno));
        return NULL;
    }

    WatchlistImpl* ptr = (WatchlistImpl*)mallocAsserted(sizeof(WatchlistImpl));
    *ptr = WatchlistImpl {
        // TODO we should ditch ArrayList here and just use realloc or whatever. Then we can store all this in
        // a struct WatchListImpl { int inotify_fd, u32 modified_file_count, FileID modified_files[] } where
        // the end of the struct is variable-length and the whole struct is heap-allocated.
        .modified_files = ArrayList<FileID>::withCapacity(1), // TODO weird arbitrary choice
        .inotify_fd = inotify_fd,
    };

    return ptr;
}

void destroyWatchlist(Watchlist watchlist) {

    int result = close(watchlist->inotify_fd);
    assertErrno(result == 0);

    watchlist->modified_files.free();
    free(watchlist);
}

FileID addFileToModificationWatchlist(Watchlist watchlist, const char* filepath) {

    assert(watchlist->inotify_fd >= 0 && "Watchlist is invalid. Did you forget to initialize it by calling createWatchlist()?");

    int watch_descriptor = inotify_add_watch(watchlist->inotify_fd, filepath, IN_MODIFY);
    // TODO treat specially the case where errno means "filepath is too long"?
    assertErrno(watch_descriptor >= 0);

    return (FileID)watch_descriptor;
}

/// You do not own the returned buffer.
/// The returned pointer is valid until one of the following occurs:
/// - the next call to `poll()` on this watchlist
/// - the next call to `destroyWatchlist()` on this watchlist.
void poll(Watchlist watchlist, u32* event_count_out, const FileID** events_out) {

    assert(watchlist->inotify_fd >= 0 && "Watchlist is invalid. Did you forget to initialize it by calling createWatchlist()?");

    watchlist->modified_files.resetSize();

    while (true) {

        // The last member of `struct inotify_event` is a variable-length `char name[]`.
        // `man 7 inotify` says that "`sizeof(struct inotify_event) + NAME_MAX + 1` will be sufficient".
        constexpr size_t event_buffer_size = sizeof(inotify_event) + NAME_MAX + 1;
        alignas(inotify_event) u8 event_buffer[event_buffer_size];

        const ssize_t bytes_read_count = read(
            watchlist->inotify_fd,
            &event_buffer,
            sizeof(inotify_event) + NAME_MAX + 1
        );

        if (bytes_read_count < 0) {
            // EAGAIN means that the fd is marked nonblocking and the read would block. See `man 2 read`.
            if (errno == EAGAIN) goto LABEL_EXIT_EVENT_READ_LOOP;
            // TODO deal with EINTR?
            else assertErrno(false);
        }


        u32 event_count = 0;
        size_t buffer_pos = 0;
        while (buffer_pos < (size_t)bytes_read_count) {
            event_count++;
            inotify_event* p_event = (inotify_event*)&event_buffer[buffer_pos];
            buffer_pos += sizeof(inotify_event) + p_event->len;
        }
        watchlist->modified_files.reserveAdditional(event_count);

        buffer_pos = 0;
        while (buffer_pos < (size_t)bytes_read_count) {
            inotify_event* p_event = (inotify_event*)&event_buffer[buffer_pos];
            watchlist->modified_files.push((FileID)p_event->wd);
            buffer_pos += sizeof(inotify_event) + p_event->len;
        }
    }
    LABEL_EXIT_EVENT_READ_LOOP: {}

    *event_count_out = watchlist->modified_files.size;
    *events_out = watchlist->modified_files.ptr;
}

//
// ===========================================================================================================
//

} // namespace

