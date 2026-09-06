# Host guest-memory tests

These fixtures exercise the host's mapping/readability caches, direct-memory alias registries,
write-watch fault handling, and memory search. They model actual mapped pages and mapping lifetimes;
guest-facing allocation/API behavior belongs to HLE tests, and GPU cache publication belongs to
the live compute execution fixtures. Linux write-watch tests need a real alternate-stack fault
handler so direct stores test dirty tracking rather than an explicit-notification substitute.
