#ifndef _FILE_WATCH_HPP
#define _FILE_WATCH_HPP

// #include "types.hpp"

namespace filewatch {

//
// ===========================================================================================================
//

using FileID = u32;

struct FileWatchlist;

FileWatchlist createWatchList(void);
void destroyWatchlist(FileWatchlist watchlist);

FileID addFileToModificationWatchlist(FileWatchlist watchlist, const char* filepath);

void poll(FileWatchlist watchlist, u32* event_count_out, const FileID** events_out);


//
// ===========================================================================================================
//

} // namespace

#endif // include guard
