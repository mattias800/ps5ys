# Guest-input memory-safety model

prosper parses several classes of **untrusted, attacker-controllable input** entirely in the host
process: the game dump's SELF/ELF (headers, dynamic section, relocation/symbol tables), the AGC/PM4
command stream and RDNA2 shader bytecode, Gen5 resource descriptors, and guest save/param blobs. Guest
CPU code runs natively, so the risk here is **not** the guest faulting itself — it is a malformed input
driving a **host** out-of-bounds read/write or an unbounded allocation (OOM). This note records the
canonical bounding primitives and the surfaces that have been swept, so new parsers follow the pattern
and reviewers know what the invariants are.

## Canonical primitives — use these, do not hand-roll bounds

- **`self_read_ok(off, need, total)`** (`src/self/module.hpp`) — the wrap-safe bounds check:
  `need <= total && off <= total - need`. **Never** write `off + need <= total`: an `off` near 2^64
  wraps the sum to a small value and passes, then the caller reads a wild pointer. Ordering `need <=
  total` first makes `total - need` unable to underflow. This is the shape every offset+length check
  must take.
- **`clamp_table_bytes(table_file_off, declared_bytes, file_size)`** (`src/self/module.hpp`) — clamp a
  *guest-declared* table size (`DT_RELASZ`, `DT_PLTRELSZ`, a symtab span) to the bytes actually in the
  file: `min(declared, file_size - off)`, `0` past EOF. A table can't legitimately extend past the file;
  without this a huge declared size drives an enormous vector → OOM (#1219). Any guest-declared count
  that sizes a loop or a vector needs an equivalent file/pool bound.
- **`guest_readable(addr, bytes)`** — fault-safe mappedness check for a *guest virtual address* before a
  bulk read from it (used across the GPU command processor / executor / compute paths). A packet- or
  descriptor-supplied pointer is validated for exactly the bytes touched before the copy (see #1200 /
  #1202 for the two command-processor readers that were missing it).
- **`guest_writable(addr, bytes)`** — the same question for a *store*, and the one to use in front of
  one. **`guest_readable` is not a substitute:** it accepts a read-only mapping (Linux/macOS probe by
  having the kernel read the byte; Windows explicitly accepts `PAGE_READONLY`/`PAGE_WRITECOPY`/
  `PAGE_EXECUTE_READ`), so a read probe in front of a write proves the wrong property and the store
  then faults on a page the probe just called fine — the guest module's own `.text`/`.rodata` are large
  read-only mappings in the same address space. That was a real defect in #1637 as merged, fixed by
  #1654.
  **Cost, and what to do on a hot path:** `guest_writable` has a separate thread-local positive
  range cache, invalidated by the mapping/protection generation (#2387/#2389). On supporting Linux
  kernels, misses query exact covering writable VMAs using `PROCMAP_QUERY` (#3398); unavailable
  interfaces retain textual `/proc/self/maps` enumeration. Adjacent VMAs must cover the entire
  request, with no read-only region or hole, and only a complete positive query may populate the
  cache. The generation is captured **before** the evidence it tags. Neither route supplies a
  lifetime guarantee or an atomic multi-VMA snapshot. Do **not** replace it with `guest_readable`
  (wrong predicate, above). Prefer an **arithmetic answer to
  the question actually being asked**: `hle_agc.cpp`'s DCB ring registry (#1650) remembers the
  `[bottom, top)` extent each command packet was built into, so a patcher validates the pointer the
  guest hands back by range compare, and only misses fall through to the probe. Structure such a cache
  as a pure accelerator — a hit skips the probe, a miss falls back — so the safety argument rests on
  the predicate and never on the cache.
- **`rd<T>(file, off)`** (`src/self/module.cpp`) — bounded structured read: gates the `memcpy` on
  `self_read_ok` and zero-fills a `T{}` on an out-of-range offset. Every SELF/ELF header/table read goes
  through it; a raw `reinterpret_cast`/`memcpy` into `file.data() + off` is a red flag.

## Integer-overflow discipline

Every `addr + len`, `base + offset`, `count * stride`, `width * height * bpt`, and alignment round-up
(`(x + align - 1) & ~(align-1)`) over guest-controlled operands must be either computed in a wider type
(`uint64_t` accumulators for image/buffer sizes, then re-checked `> UINT32_MAX`) or guarded against wrap
(`len > UINT64_MAX - addr` before `addr + len`, as `normalize_guest_page_range` does). A wrap that
defeats a later `<=` check is the classic loader/GPU OOB.

## Descriptor dimension clamps (the load-bearing GPU invariant)

The entire GPU guest-input bounds argument rests on `decode_image_descriptor` /
`decode_buffer_descriptor` (`src/gpu/agc/agc_shader_layout.cpp`) clamping guest T#/V# fields at the source:
width/height are 14-bit fields `+1` → **≤ 16384**, depth 13-bit `+1` → **≤ 8192**, format masked to
9 bits, `num_components ∈ {0..4}`, buffer `size_bytes` computed in `uint64_t` and clamped to
`0xFFFFFFFF`. Downstream code multiplies these into host allocation/copy/detile bounds and rejects
`> UINT32_MAX` / `> kMaxComputeImageBytes` — sound **only** because the dimensions can't exceed those
masks. `tests/gpu/agc/test_descriptor_clamp.cpp` pins these ceilings so a mask-widening refactor fails loudly.

## Swept surfaces (systematic adversarial hunts)

Each of these was audited by an adversarial memory-safety hunt (find OOB read/write + integer-overflow-
in-bounds, rule out upstream guards). All CLEAN on current master except the edge-path fixes noted:

| Surface | File(s) | Status |
|---|---|---|
| SELF/ELF header + segment parse | `src/self/module.cpp` (`SelfHeader`/`Phdr`/`va2foff`/ELF-magic scan) | CLEAN — all `rd`-bounded; scan loop-bounded `i+4 < size` |
| Relocation-apply write path | `src/self/module.cpp` `apply_relocations` | CLEAN write path (double-bounded `img.at` + `p+8` guard); **#1219** clamped unbounded table sizes (OOM) |
| Multi-module linker | `src/loader/linker.cpp` | CLEAN; **#1198** init-array `p+8` guard |
| Loader bounds primitives | `src/self/module.{hpp,cpp}` | **#1198** `self_read_ok` (replaces wrap-prone `off+need<=size`) |
| Host image map + ABI stubs | `src/host/image/exec_image_linux.cpp` `map_image`/`install_stubs` | CLEAN — `img.mem`-sized copy; stub region = import-count×stub_size (count clamped by #1219) |
| Guest memory-management HLE | `src/hle/memory/hle_kernel_mem.cpp` | CLEAN — WriteAddress fixed-8B `process_vm_writev`, VirtualQuery clamped, dmem allocator pool-bounded, every `addr+len` guarded |
| GPU command processor / PM4 | `src/gpu/pm4/command_processor.cpp` | **#1200** SetRegsIndirect `guest_readable` guard; **#1202** WriteData null-src stale-tail `memset` |
| GPU executor / shader decode | `src/gpu/execute/gpu_executor.cpp` | CLEAN (canonical wrap-safe form throughout); **#1210** diagnostic-dump OOB clamp |
| Compute dispatch sizing | `frontends/shared/live/live_compute.cpp` | CLEAN |
| RDNA2→SPIR-V recompiler | `src/gpu/recompiler/rdna2_to_spirv.cpp` | CLEAN — literal/branch/register/CFG all upstream-bounded |
| Tile / detile swizzle | `src/gpu/texture/tile.cpp` | CLEAN — every tiled access guarded `if (tiled + bpe <= tiled_bytes)`; mismatch drops, never OOB |
| Descriptor decode | `src/gpu/agc/agc_shader_layout.cpp` | CLEAN — see the dimension-clamp section above |
| Resource/descriptor layer, GPU capture parser | `src/gpu/*` | CLEAN |
| HLE getters (audio/service/http/json) | `src/hle/*` | CLEAN — output params fully initialized |
| SELF/save path traversal | `src/hle/fs/hle_file.cpp` | **#1204** savedata `dirName` guard; **#1206** `translate()` `..` normalization |
| `__cxa_guard`/`call_once` recursion | `src/hle/libc/hle_libc.cpp` | **#1196** recursion self-deadlock break |

## Adding a new guest-input parser?

1. Route every read of the untrusted buffer through a bounded accessor (`rd`/`self_read_ok`/
   `guest_readable`); never index the raw buffer without a loop bound proven `≤ size - N`.
2. Clamp every guest-declared count/size that drives a loop, vector, or allocation to the file/pool
   extent (`clamp_table_bytes` or an equivalent) — bounded reads still OOM if the *count* is unbounded.
3. Use the wrap-safe `total - off >= need` form and `uint64_t` accumulators for products.
4. Add a discriminating unit test (verify it fails/faults when the guard is reverted), and — for a
   non-trivial parser — get an adversarial hunt over it before merge.
