#ifndef _FILE_WATCH_HPP
#define _FILE_WATCH_HPP

// #include "types.hpp"

namespace filewatch {

//
// ===========================================================================================================
//

using FileID = u32;

struct WatchlistImpl;
using Watchlist = WatchlistImpl*;

/// Returns NULL on failure.
Watchlist createWatchlist(void);
void destroyWatchlist(Watchlist watchlist);

FileID addFileToModificationWatchlist(Watchlist watchlist, const char* filepath);

void poll(Watchlist watchlist, u32* event_count_out, const FileID** events_out);


//
// ===========================================================================================================
//

} // namespace

#endif // include guard
