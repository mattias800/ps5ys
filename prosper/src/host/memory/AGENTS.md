# `src/host/memory` — what the HOST knows about the guest's address space

This folder holds the host-side view of guest memory: which ranges are mapped and readable, which
pages are being watched for writes, and how to look through the whole space for something. It is
deliberately *below* `src/hle/memory`, which is the opposite direction — that folder implements the
guest-facing `sceKernel*` memory API and decides what gets mapped; this one answers questions about
whatever ended up mapped, without any opinion on why.

- **`guest_memory_map.hpp`** — the readable/writable range caches every consumer consults before
  dereferencing a guest pointer, plus the mapping *generation* counter that invalidates them. The
  generation is the load-bearing part: every change to guest page protection, from anywhere in the
  tree, must advance it or a cache serves a stale answer about a page that has since been revoked.
- **`guest_memory_query.{hpp,cpp}`** — uncached, exact-cover writable VMA queries where the host
  supports them. It reports unavailable separately from non-writable; cache policy and mapping
  generations remain with its consumers.
- **`guest_write_watch.{hpp,cpp}`** — write-protection-based dirty tracking for guest pages, and the
  direct-memory write trace that attributes a guest write to the module and RIP that made it. Host
  writes into guest memory (file reads, DMA-shaped producers) must bracket themselves with the
  notify/notify-done pair here, or they either fault or leave a range recorded as in flight forever.
- **`guest_memory_search.{hpp,cpp}`** — "where else in the guest's address space do these exact bytes
  appear?". Used when an address stops holding what it held and the question becomes whether the
  guest *moved* the data or *consumed* it, which need opposite fixes. Its pure half owns the chunking
  and overlap and is unit-tested; only the `/proc/self/maps` enumeration is POSIX-specific.

**What belongs here:** anything that reads or classifies guest memory from the host side and has no
guest-facing ABI of its own. **What does not:** the allocators and mappers themselves
(`src/hle/memory`), the fault handler and image mapping (`src/host/image`), and anything that
interprets guest memory as a *resource* — a texture, a descriptor, a command buffer — which is
`src/gpu`'s job.

The recurring hazard in this folder is that its instruments are consulted to publish **negatives**
("nothing wrote it", "it is not there"), and a negative from a half-armed or narrowly-scoped
instrument is indistinguishable from a real one. So every result type here carries its own scope —
what was actually looked at, what could not be read, whether a cap or budget cut the search short —
and callers are expected to print it beside the verdict rather than quoting the verdict alone.
