#include "host/memory/guest_memory_query.hpp"

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace prosper::host {

GuestWritableQueryResult query_guest_writable_range(uint64_t begin, uint64_t end) {
#if defined(__linux__) && defined(PROCMAP_QUERY)
    if (begin >= end) return {GuestWritableQueryStatus::NotWritable};
    // Reopen per probe: retaining /proc/self/maps across fork would retain the old
    // process's mm. Request neither path names nor ELF build IDs.
    const int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return {};
    const auto result = detail::query_contiguous_writable_range(begin, end,
        [fd](uint64_t cursor) -> GuestWritableQueryResult {
            procmap_query query{};
            query.size = sizeof(query);
            query.query_flags = PROCMAP_QUERY_VMA_WRITABLE;
            query.query_addr = cursor;
            // No COVERING_OR_NEXT: an unmapped or read-only gap refuses the
            // request, even when a later VMA is writable.
            if (ioctl(fd, PROCMAP_QUERY, &query) != 0)
                return {errno == ENOENT ? GuestWritableQueryStatus::NotWritable
                                       : GuestWritableQueryStatus::Unavailable};
            if (!(query.vma_flags & PROCMAP_QUERY_VMA_WRITABLE))
                return {GuestWritableQueryStatus::NotWritable};
            return {GuestWritableQueryStatus::Writable, query.vma_start, query.vma_end};
        });
    close(fd);
    return result;
#else
    (void)begin;
    (void)end;
    return {};
#endif
}

} // namespace prosper::host
