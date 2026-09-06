#pragma once

#include <cstdint>

namespace prosper::host {

enum class GuestWritableQueryStatus { Writable, NotWritable, Unavailable };

struct GuestWritableQueryResult {
    GuestWritableQueryStatus status = GuestWritableQueryStatus::Unavailable;
    uint64_t begin = 0;
    uint64_t end = 0;
};

namespace detail {
// Query must return the writable VMA COVERING cursor, not the next writable VMA.
// No partial result escapes: every byte of [begin, end) must be covered before a
// positive range can be published. Successive OS queries are not an atomic snapshot.
template <class Query>
GuestWritableQueryResult query_contiguous_writable_range(
    uint64_t begin, uint64_t end, Query&& query) {
    if (begin >= end) return {GuestWritableQueryStatus::NotWritable};
    uint64_t cursor = begin;
    uint64_t span_begin = begin;
    while (cursor < end) {
        const GuestWritableQueryResult region = query(cursor);
        if (region.status != GuestWritableQueryStatus::Writable)
            return {region.status};
        if (region.begin > cursor || region.end <= cursor)
            return {GuestWritableQueryStatus::Unavailable};
        if (cursor == begin) span_begin = region.begin;
        if (region.end >= end)
            return {GuestWritableQueryStatus::Writable, span_begin, region.end};
        cursor = region.end;
    }
    return {GuestWritableQueryStatus::NotWritable};
}
} // namespace detail

// Linux binary covering-VMA queries, where supported. Unavailable tells callers to
// retain their portable OS probe. No cached fd, permission result or mapping epoch.
GuestWritableQueryResult query_guest_writable_range(uint64_t begin, uint64_t end);

} // namespace prosper::host
