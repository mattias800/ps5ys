# Resource binding — the front-half ↔ recompiler contract

> **Current note (2026-08-11): the contract and its front-half materialization are implemented.** GTA V's
> six-stage runtime-selected descriptor-array lift now covers capability negotiation, reflection,
> SPIR-V indexing, descriptor layouts/writes and pipeline-cache identity. The black world at routed
> gameplay entry persisted after that lift, so #2412/#2481 track the current exact compute terminals.
> The staged plan below is retained to explain the contract, not as an open implementation checklist.

**Purpose.** Unblock the format/descriptor-dependent shader memory instructions — `s_buffer_load_*`
(multi-buffer uniforms), `buffer_load_format_*` (vertex fetch), and `image_sample`/`image_load`
(textures) — *correctly*, without guessing formats. The seam is `src/gpu/resources/shader_resources.hpp`.

## Why a contract (the core problem)

To translate `buffer_load_format_xyzw v[..], vaddr, s[8:11], …` the recompiler must know the vertex
attribute's **data format** — float32? unorm8? snorm16? — to emit the right conversion. A float32
attribute is a raw dword load (no conversion in our raw-32-bit-VGPR model); a unorm8×4 attribute is a
`load dword → unpack 4 bytes → /255.0`. **That format lives in the V# descriptor the game builds**,
which the recompiler never sees directly (it's in memory the game loads via `s_load`). So the
recompiler cannot, on its own, translate a format load correctly. It must be *told* the formats.

**Division of labor:**
- **Front-half (agent 2 / the AGC HLE):** knows the game's real GPU resources. From the shader's
  `user_data` / SRT (already traced — `ShaderUserData`: `direct_resource_offset`,
  `sharp_resource_offset[4]`, counts) and the game's bound resources, it reads the **V#/T#/S#
  descriptors** (base address, stride, `DFMT`/`NFMT`, dims, …) and produces a `ShaderResourceTable`.
- **Back-half (recompiler, me):** parameterized by that table. While translating a memory op it
  resolves *which* resource the op targets (see "provenance" below), emits the correct binding +
  format conversion, and records the binding layout.
- **Pipeline (back-half):** binds `size` bytes at each resource's `gpu_addr` (unified guest memory)
  to descriptor-set 0, `binding` — real game data flowing to the shader.

The result: real shaders get **real inputs** (turning the current near-black demo output real), and
vertex/texture shaders recompile correctly.

## Descriptor provenance — how a memory op maps to a resource

A shader uses a resource in two steps:
```
s_load_dwordx4  s[8:11], s[srt_ptr], 0x20      ; load a V# descriptor from user_data offset 0x20
buffer_load_format_xyzw v[0:3], v1, s[8:11], 0  ; use it to fetch a vertex attribute
```
So the recompiler tracks, per SGPR, **which SRT/user_data byte offset a descriptor came from**:
- On an `s_load*`/`s_buffer_load*` whose SBASE is the shader's user_data pointer, tag the destination
  SGPRs with `srt_offset = <the load's offset>`.
- On a `buffer_load_format_*` / `image_*` / `s_buffer_load_*`, read the tag off its SRSRC/SBASE SGPRs
  and look up `table.by_srt_offset(tag)` → the `ShaderResource` (format, binding, …).

This is a small SGPR-provenance side-table in the recompiler (analogous to the existing VGPR/SGPR SSA
maps). It keys the abstract "resource N" to the concrete descriptor the front-half described. The
front-half fills `ShaderResource::srt_offset` with the same offset so the two sides rendezvous.

### Two provenance modes (INDIRECT vs DIRECT)

Not every descriptor is loaded in-shader. Sony resources come in two flavours, and a `ShaderResource`
sets whichever key matches (leaving the other `0xFFFFFFFF`):
- **INDIRECT (`srt_offset`)** — the shader `s_load_dwordx4`s the V# from its user_data/SRT (above).
  Constant buffers (`s_buffer_load`) are typically this. Recompiler tags the load, resolves by offset.
- **DIRECT (`sgpr_base`)** — the driver places the V# *straight into the user-data SGPRs* at launch
  (Sony "direct" resources). **Vertex-buffer descriptors are this** — there is no in-shader load to
  tag. The recompiler resolves a memory op by matching its SRSRC/SBASE SGPR index to `sgpr_base`
  (`by_sgpr_base`). The recompiler must know the shader's user-data→SGPR layout at entry; the
  front-half provides `sgpr_base` = the SGPR the V# occupies.

So the recompiler resolves a memory op's descriptor by: (1) an `s_load` provenance tag → `by_srt_offset`,
else (2) the SRSRC/SBASE SGPR index → `by_sgpr_base`.

`DataFormat` is decoded from the descriptor's `DFMT`/`NFMT` (e.g. `DFMT=32_32_32_32,NFMT=FLOAT` →
`Float32`, `num_components=4`; `DFMT=8_8_8_8,NFMT=UNORM` → `Unorm8`, `num_components=4`). That decode
lives front-half (it owns the descriptor bit layout); the recompiler only consumes `DataFormat`.

## Binding scheme

- Descriptor set 0. Bindings assigned by the front-half when it builds the table (stable per shader).
  Convention: constant buffers first, then vertex buffers, then textures, then samplers — but the
  recompiler only relies on the `binding` value in each `ShaderResource`, not the order.
- The recompiler's existing single constant buffer (binding 2 in the compute shell) generalizes to
  "the `ShaderResource` with that binding."

## Staged implementation (each stage its own execution-verified commit)

1. **Multi-buffer `s_buffer_load`** (generalize today's single-cbuf model). Provenance-track the V#
   SGPRs → `by_srt_offset` → per-resource binding. Test: two constant buffers, distinct contents.
   **DONE** (kernel 22 — provenance routes two `s_buffer_load`s to bindings 2 & 3).
2. **`buffer_load_format_*` — float32 fast path.** For `DataFormat::Float32` (positions, most
   attrs) a format load is a raw dword load — reuse the MUBUF path, keyed to a `VertexBuffer`
   resource. This is what unblocks the game's real vertex shaders (op 0x0/0x3 dominate). Test: a
   float32 vertex fetch through the table. **DONE** (kernel 23 exec-diff; `vertex_fetch_render`
   renders real buffer-sourced geometry through a bound VkPipeline).
3. **`buffer_load_format_*` — packed conversions.** `Unorm8`/`Snorm16`/… : load dword → unpack →
   normalize (bitfield-extract → convert → /255|/32767, SNORM clamped to ≥ −1.0; Float16 via
   `UnpackHalf2x16`). Test: unorm8×4 → 4 floats in [0,1]. **DONE** — `unpack_norm`/`unpack_half`
   cover Unorm8/Snorm8/Unorm16/Snorm16/Float16; kernels 24 (unorm8×4) & 25 (snorm16×2 + clamp)
   verify the exact numbers. Integer sub-dword formats (Uint8/Sint8/…) are still rejected (no
   integer-attribute path yet) rather than mis-normalized.
4. **MIMG `image_sample`/`image_load`.** Bind a Vulkan sampled image + sampler from the T#/S#;
   emit `OpImageSampleImplicitLod`. Test: sample a known 2×2 texture. **DONE** — combined image+sampler
   (`OpTypeImage`/`OpTypeSampledImage`); `image_sample` (0x20, implicit LOD), `image_sample_lz` (0x27,
   LOD 0) / `image_sample_l` (0x24, explicit LOD) via `OpImageSampleExplicitLod`, `image_load` (0x00)
   via `OpImageFetch`; T# resolved through the same SRSRC provenance as vertex buffers (SMEM x8 tags it).
   Tests: `texture_sample_render` (2×2 texel, u/v routing + LOD variant + fetch) and `textured_interp_
   render` (VS-interpolated UVs → sample, the full textured-draw path). 2D float non-NSA; other dims /
   NSA / gradient / compare-shadow variants deferred (rejected, not faked).

## Beyond the staged plan (also DONE this line of work)

- **MUBUF stores** — `buffer_store_dword/x2/x4` + `buffer_store_format_*` (raw/Float32), with a real
  **EXEC-predicated conditional store** (selection-merge on the per-lane EXEC bool) + `robustBufferAccess`.
- **VINTRP** — pixel-shader attribute interpolation: VS `EXP PARAM_n` → Output varying, PS
  `v_interp_p1/p2/mov` → interpolated Input varying (deferred EntryPoint so varyings join the interface).
- **`s_cbranch_execz` guard-to-`s_endpgm`** linearization; **SCC** (`s_cmp`/`s_cselect`).

**Historical status at the end of the staged plan:** the recompiler covered the full shape of a real
textured, interpolated, buffer-reading/writing draw, while the runtime resource-table producer was the
next missing piece. That producer has since landed: graphics and compute discovery recover direct,
indirect and runtime-selected V#/T#/S# descriptors, materialize their resources, and pass a reflected
table to the recompiler. A raw table-less coverage metric still cannot classify a table-dependent
instruction; use a live or resource-bearing replay result instead.

Stage 2 was the high-value unlock (real VS recompile → real VS+PS frames from the game). Stages 1–3
depend only on this contract + the front-half filling the table for constant/vertex buffers; stage 4
adds textures.

## Historical front-half deliverable (implemented)

The implemented producer walks the shader's `user_data`/SRT, reads each V#/T#/S# descriptor, decodes
format/dimensions/base/stride, assigns bindings, and records the direct SGPR or indirect SRT provenance
used by the instruction. It then makes the referenced guest bytes or image available to the backend
and passes the resulting `ShaderResourceTable` into recompilation. Runtime-selected arrays extend this
same contract with reflected descriptor count and dynamic-index metadata; they are not a parallel path.

## Storage-image alias seed authority

Live compute resolves each dispatch's exact storage-image alias groups before preparing their
canonical images (`frontends/shared/compute/storage_image_alias_plan.hpp`). Access is a group
property: any readable or atomic member prohibits seed omission and destructive poison proving.
The plan follows the existing late-fold identity, including view shape, capture backing and DCC;
it does not change which descriptors actually share a Vulkan image. Sampled images remain separate.

Cached coverage verdicts include the sorted group binding set. Moving an intact group does not
change that identity, but joining or splitting does: two writers may cover an entire image together
while neither does alone, and a previously inactive owner may gain a writer. Both proof lookup and
publication use the same member-aware key. These are seed-safety constraints, not a static proof
that a shader's coverage is invariant under arbitrary input changes (#3391).

### Ruled out: per-binding seed safety

- **A write-only owner's reflection is sufficient to omit seed data, provided untouched image
  texels are restored afterwards:** falsified by a race-free storage-image read into an independent
  SSBO. Restoring the image cannot repair values already consumed by another output. Reverting
  only the access union produces seven exact-read failures in cold, warm and newly joined cases;
  reader-first controls pass (#3391).
- **Code, owner binding and extents identify a reusable coverage verdict regardless of alias
  membership:** falsified by fixed-code, fixed-extent split/join fixtures. Reverting only membership
  in the proof key produces two stale-Full and two stale-None failures (#3391).

## Ruled out

Cross-title falsifications about **how the user-data / SH register block reaches a stage**. One line
per dead hypothesis, the evidence that killed it, and where that evidence lives. Do not re-derive
these without contradictory new evidence.

The umbrella is **#305** — a graphics stage resolves garbage descriptors exactly when the user-data
block the guest most recently programmed is **larger** than the bound pipeline's user-SGPR window
(`SPI_SHADER_PGM_RSRC2_{GS,PS}.USER_SGPR`, which equals the shader's own `user_data_range_end`). The
shader then dereferences a V# `num_records`/`dword3` tail as a pointer — the `0x0004dfac…` /
`0x0001d22c…` constant family — the const-fold correctly refuses to invent a descriptor, and the draw
is skipped fail-visibly. Confirmed on Nikoderiko (`PPSA23760`, #1607), DOLL / Dragon Quest VII
(`PPSA17942`) and **Stray (`PPSA02101`, #3137)** — all three UE4; **not** reproduced on The Pathless
or The Plucky Squire.

Stray is the cleanest reproduction so far because both halves are visible in one `PROSPER_DYNTRACE_FAIL`
replay. Its 221-dword title-screen VS is bound with `RSRC2_GS.USER_SGPR = 8 = user_data_range_end`
while `PROSPER_UDPROV` shows the draw's own bind writing **twelve** contiguous dwords in one direct
`SET_SH_REG` a few packets earlier, so dwords 8..11 — two mapped 64-bit pointers — are outside the
window the shader can see. The shader's `s_load_dwordx4 s[16:19], s[14:15]` therefore reads user
dwords 6:7, which hold the `num_records`/`dword3` tail of the preceding V#: `0x0004dfac00000001`,
the same constant family. The fold reports `addr … unreadable`, every later scalar is `ok=0`, and
the vertex fetch that consumes the descriptor is left unresolved.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| A vertex-fetch `[recompile-reject] mode=unresolved-operand` on `buffer_load_format_* … idxen` means the recompiler lacks that lowering, **or** that the stage arrived with no resource table (`allow_smem = rt != nullptr`) | **Both falsified on Stray (`PPSA02101`).** MUBUF `0x0`–`0x3` are lowered, and the disposition line says which arm fired: `[buf-op] … MUBUF op=0x3 n=4 … rt=1 reject-unresolved`, i.e. the table was present with five resources and only the SRSRC failed to resolve. The soffset was not the cause either — the two failing fetches carry `VCC_LO`/`VCC_HI` as SOFFSET, which `operand_bits`' vertex-stage branch resolves, and in any case the reject happens *before* the address is formed. The real cause is the #305 user-data window above. **Read `[buf-op]`'s `how=` and `[mubuf-unresolved]`'s `rewritten=`/`pc_res=` before concluding anything about a MUBUF reject from the census line**; the format path only started printing those two fields in #3137's PR, which is why this pair of hypotheses looked plausible. | #3137 |
| The `[udcand]` "implied seed" search (the one window offset at which every declared direct pointer becomes mapped) can be turned into behaviour: re-seed the block from it when the resolved header misfits | **Falsified, and it is the most attractive wrong fix for #305 — it is loud, self-validating and still wrong.** On Stray's 221-dword VS the search reports `seeded=0 implied=4`, and at seed 4 both declared direct pointers do become mapped. But the same shift moves `s[8:11]` onto a `stride=16 / num_records=1` V# — 16 bytes — while the shader's own `s_buffer_load_dwordx4 s[20:23], s[8:11], 0x30` reads at byte 48, and its `s_buffer_load_dwordx8 s[0:7], s[8:11]` reads 32. At seed 0 the same registers decode as `stride=8 / num_records=376` (512 bytes), both accesses are in bounds, and the fold prints the loaded bytes as plausible floats. So the pointer-mapping criterion is satisfiable by an offset the shader's own buffer accesses refute: it tests a property the block has by coincidence, not the layout. Any re-seeding rule must be checked against the *shader's* accesses, not only against pointer mappedness. | #3137, #305 |
| A writable T#/U# (`sharp_resource_offset[1]`, size=0) that `build_shader_resources` skips is still "handled by the dynamic descriptor fold" (`resolve_dynamic_fetch`'s SrtUse mechanism in `gpu_executor.cpp`), so skipping it in the static builder loses nothing | **Falsified for the common DIRECT case.** That mechanism resolves a T# only by tracing an in-shader `s_load` (table-loaded) or by proving the SRSRC's 8 SGPRs are an *untouched/copied entry seed* — it has no path that consults `sharp_resource_offset[1]` at all. A T# the driver places straight into the entry user-data SGPRs (the ordinary case) still needs THIS array to know a descriptor lives there in the first place; nothing else can discover it. Both `build_shader_resources` loops over category [1] previously skipped it — the buffer loop's `!s.size()` guard (correctly, to avoid misreading an 8-dword T# as a garbage V#) and the texture loop below it (which only ever walks category [0]) — so it fell through both and never entered `table.resources`. Measured on one title: 58 of 86 writable slots (PROSPER_SHARPLOG), with one stage showing 4 declared writable T#s and a completely empty resource table, rejected downstream as `[mimg-unresolved]`. Fixed by decoding it in place with the same pipeline the read-only texture loop uses, classified `ResourceClass::StorageImage`. | #3128 |
| A scalar `S_BUFFER_LOAD`'s bound is **independent** of the V#'s vector footprint — `M_SIZE` = one dword when `STRIDE == 0`, else `NUM_RECORDS` **dwords** | **Falsified in both branches.** The scalar bound IS the descriptor's byte footprint, `(stride ? stride : 1) * num_records` — exactly `decode_buffer_descriptor::size_bytes`. STRIDE scales the bound; it never converts it into a record or dword count. `STRIDE == 0`: the dword reading makes every ordinary constant buffer one dword long, and *The Messenger* — first level checked against PS5 hardware — rendered **111,118 consecutive byte-identical fully black frames** with *Dead Cells* collapsing identically (both guards: 0 qualifying frames, 0 pixel changes, 0 structural matches). `STRIDE != 0`: GTA V's `0x413ced900` walks a `STRIDE=120`/`NUM_RECORDS=5` descriptor table with `s_mul_i32 vcc_hi, s9, 0x78` + `s_buffer_load_dwordx4 …, vcc_hi offset:0x8` for `s9` in 0..4; a NUM_RECORDS-dword bound admits 20 of those 600 bytes, so four of five records loaded an all-zero V# and the shader's bounding-box reduction read zeros for every vertex. Capsule A/B: **269** dispatches resolved every record to its real vertex buffer at `0a0a294a`, versus **0** at `8ae101c5`. **The trap is the source, not the reasoning:** the RDNA2 guide's own §7.2.1 prints `m_size = (m_stride == 0) ? 1 : m_num_records`, dropping the stride factor, while its buffer-descriptor table (§8.1.8) says NUM_RECORDS is in units of stride when strided; the RDNA3 (§8.1.2, OOB in §8.4.1) and RDNA4 (§8.1.2) guides both print the corrected product. Do not re-derive this bound from RDNA2 §7.2.1 alone. The rule reached master because it was introduced in `29ec3cd4` **guarded to empty descriptors**, where its `STRIDE != 0` branch is trivially zero and was never exercised, and `8ae101c5` then removed the guard and promoted the untested branch. | #2528 |
| `texture_sample_render` returning the expected red pixel meant its UINT storage-image module was compatible with the bound `R8G8B8A8_UNORM` view | **Falsified.** Vulkan validation reports `VUID-vkCmdDraw-format-07753` at the draw: the module's Sampled Type is UINT while the view is floating-point/normalized. RADV happened to return the expected pixel under that undefined contract, so the old pixel assertion could not diagnose it. Reflection now carries Float/Uint/Sint explicitly; graphics expands the portable UINT ABI to `R32G32B32A32_UINT`, and the backend rejects unknown or mismatched classes before Vulkan. A defect mutation restores the exact VUID and fails the contract check. Live `build_R` coverage now also proves that two same-span references execute one compact-guest conversion plus one representation-preserving cache hit, that formatless raw-uvec4 and exact `R32ui` views over one T# remain distinct identities, and that a short padded tiled backing fails before detile instead of becoming zero-extended content. | #1713 |
| A sampled resource that must be a plain 2D surface can be identified by requiring `img_dim == 1` | **Falsified across the AvPlayer path.** A guest may DECLARE a byte-identical surface as `DIM=2D_ARRAY` with one layer, which `shader_resource_uses_ordinary_2d_image` already treats as an ordinary 2D image and which the sampled-texture path already admits in its padded-row read, its source-address computation and the `VK_IMAGE_VIEW_TYPE_2D` view it creates. R-Type Delta (`PPSA26414`) declares BOTH NV12 movie planes that way; the AvPlayer chroma test's `img_dim == 1` clause therefore rejected a real chroma plane and dropped it into the legacy narrow coverage broadcast, which made the shader's V equal its U and collapsed the whole opening movie onto one green<->magenta chroma axis with luma, detail and geometry all still correct. Test one-layer-ness (`depth == 1`, no layer stride, no layer mip offset), not the declared DIM — and test it for **equality**: the descriptor decoder emits `depth = LAST_ARRAY - BASE_ARRAY + 1` for an array type and **zero** when `LAST_ARRAY < BASE_ARRAY` (`agc_shader_layout.cpp`), so `depth == 0` is a malformed inverted array range, not a single layer, and must keep failing visibly. | #2005, `frontends/shared/media/avplayer_plane_policy.hpp` |
| An NV12 plane pair can be identified by requiring the luma plane to END at the chroma plane's address (or at the 64 KiB boundary before it) | **Falsified by a title that stages its own planes.** Memory adjacency is real evidence when the *producer* stages the buffer — AvPlayer writes one contiguous NV12 frame and R-Type Delta's two descriptors are cut straight out of it — but a title that drives `sceVideodec2` receives the NV12 in its own decode ring and then stages the planes wherever it likes. *Tales of Graces f Remastered* (`PPSA19991`) samples a `2048x1088` R8 luma plane and a `1024x544` RG8 chroma plane **0x111000 bytes past its end**, and that gap alone sent every opening-movie frame down the legacy narrow coverage broadcast: `corr(Cb, Cr) = +0.9999` on movie frames against `+0.12` on title-screen frames from the same run. What makes the pair an NV12 pair is the GEOMETRY — one luma texel per chroma byte across the row, two luma rows per chroma row, the same physical row pitch, both linear, both one-layer 2D, both `Unorm8`, and both bound by **the same draw**. Keep adjacency as a stronger, separately-reported verdict; do not keep it as the requirement. The one memory relation that still disqualifies a candidate is OVERLAP. | #2731, `frontends/shared/media/avplayer_plane_policy.hpp` |
| A decoded video frame's two planes are LINEAR surfaces | **Falsified.** *Sonic Origins* (`PPSA05325`) stages its decoded `3840x2160` NV12 with both planes **GPU-tiled** (`SW_64KB_S`, `tile_mode 9`), declared as one-layer 2D arrays, and the chroma plane's `tile_mode != 0` rejection produced the identical collapse (`corr(Cb, Cr)` `+0.998`..`+0.9999` on movie frames, `-0.63` on its own SEGA logo in the same run). A tiled pair has no row pitch to reason about, so neither the HLE pitch registry nor the resolved-pitch comparison applies — but its padded **tiled** size is exact and lands precisely on the second plane (`0x870000` for `3840x2160` at 1 B/texel, against `0x7e9000` tight; four allocations out of four), so the tiled route keeps adjacency as a requirement. Note the second half: the renderer's native RG8 upload is a straight copy with **no de-swizzle**, so admitting a tiled plane to the classifier without also gating `native_rg8_sampled` on `tile_mode == 0` would upload micro-tile order as scanline order — a woven picture, worse than the collapse. | #2731, `frontends/shared/live/live_renderer.cpp` |
| A single-component plane's `DST_SEL` can be constrained to R/0/1, as a free extra discriminator for the sibling luma plane | **Falsified while writing the #2731 fix, by the project's own render tests.** All three titles measured live declare their luma plane `(R,0,0,1)`, which makes the constraint look free — but GFX10 does not tie `DST_SEL` to a format's component count, and the identity `(R,G,B,A)` remap is an ordinary descriptor for a one-component surface (the hardware returns 0 for the absent channels and 1 for alpha). Two existing arms in `tests/misc/test_gpu_capture_render.cpp` build their luma fixture with the chroma swizzle and went red immediately. Rejecting that shape would have silently re-created this very collapse on whichever title spells its luma T# that way. | #2731 |
| `VdecOutput::format` being left 0 misdescribes the chroma layout to the guest, and is the leading candidate for the collapsed-chroma cast | **Falsified.** `format` was 0 before and after the #2731 fix — untouched, and still the documented open item from #2571 — while the movies of both affected titles went from `corr(Cb, Cr) = +0.999` to the uncorrelated range. The field the guest was misreading was never in `VdecOutput`; it was the renderer's own classification of the T# the guest then built. A `PROSPER_VDEC2_FORMAT` sweep would have found nothing, whatever discriminator it was judged by. | #2731, #2571 |
| A `[mimg-unresolved]` reject means the shader's descriptor could not be RESOLVED | **Falsified.** The line reports that no `ShaderResource` matched the op's pc or SRT key — which also happens when the front half resolved the descriptor perfectly and then **refused to materialize it**. Sonic Racing: CrossWorlds (`PPSA08804`) drops two pixel pipelines this way: `PROSPER_DYNTRACE_FAIL=1` shows the const-fold recovering the same 4K T# on both, once from the entry user data and once through a descriptor table, after which all three materialization paths discard it for `d.base_array != 0`. That guard (22f7b647) was a deliberate fail-closed measure taken while the per-slice byte stride was unmodelled; #5fa3f25d modelled it in `image_base_level_view` for cube faces and 2D-array slices and **did not lift the three call-site guards**, so a layered T# never reached the code that could resolve it. Refuse a non-zero `BASE_ARRAY` only for the types whose slice origin is still unmodelled — 1D_ARRAY, 2D_MSAA_ARRAY, a 3D UAV view — via the one shared predicate `image_descriptor_reject_reason`, which now names the reason. **Before concluding anything about provenance from an unresolved-descriptor reject, replay the failing stage with `PROSPER_DYNTRACE_FAIL=1`**: "the front half never produced this resource" and "it produced it and dropped it" are indistinguishable from the reject line, and point at different files. | #1895, `agc_shader_layout.cpp`, `gpu_executor.cpp` |
| A `[mimg-unresolved]` line whose `srt_tag`, `key_res`, `pc_res` and `alias_res` are ALL empty is evidence that the descriptor is absent from the frame | **Falsified twice over — the hypothesis names the wrong four fields, and one of the four it names carries real information.** The fields whose nulls are FORCED are `srt_tag`/`key_res` (the s_load-tag and the SRT-key lookup it gates) and `ud_alias`/`alias_res` (the #1773 copy alias and the lookup it gates): `sreg_srt` and `sreg_ud_alias` populate only through `record_scalar_write`, which is also what inserts into `sreg_written`, so when the line says `written=0` — the shader never wrote the SRSRC range, i.e. the descriptor is entry user data by the shader's own construction — *neither of those two routes can fire* and their four fields restate `written=0` rather than measuring anything. **`pc_res` is NOT among them**, and the hypothesis' inclusion of it is the second error: `pc_res` is `by_fetch_pc`, keyed by the instruction with no register-state guard, and it is the resolver's FIRST route — its null is a real measurement, and separating its two possible causes is what a `PROSPER_DYNTRACE_FAIL=1` run was later needed for. The route that actually ran is `by_sgpr_base(SRSRC)`, and its outcome was **the one thing the line did not print**. Measured on Stray (`PPSA02101`) 2026-09-03, 5 sites in one title-screen run: 4 report `written=0`, and in **3 of those 4** the table held a resource at exactly the requested SGPR the whole time — a `ConstantBuffer` — which the resolver discarded on class without a word in the log. The remaining `written=1` site is the internal control: its `srt_tag=0x20` field populates precisely because its route did run. The line now prints `sgpr_res=<class>` and `need=<sampled\|storage\|either>`. Do not read a null quadruple as absence; read `sgpr_res` and `written`. This also corrects one sentence in #3126's own thread — "the resolver correctly falls through to `by_sgpr_base(8)`, and that is what returns null": the fall-through half is right, the null half was never observable, and on the vertex site it is false. Recorded as **instrument trap 254** in `GAME_COMPAT_ORCHESTRATION.md`, because the shape generalises: a diagnostic that enumerates mechanisms, alongside a field saying which mechanism applies, must be read by first asking whether the enumerated ones CAN fire under that mode. | #3126, #1634, `rdna2_emit_alu.cpp` |
| A resource printed by `[mimg-unresolved]`'s `available:` list as `srt=0xffffffff` has no lookup key | **Falsified.** The old printer chose ONE key to show: `s%u` when `sgpr_base` was set, otherwise `srt=0x%x` — so a resource keyed only by `fetch_pc`, which is exactly how the const-fold publishes a seed- or table-recovered T#, rendered as `srt=0xffffffff` and read as unkeyed. Four of the nine resources in Stray's failing pixel stage printed that way while being properly keyed. The list now prints every key a resource carries (`s8`, `srt=0x50`, `pc=69`) and the literal word `unkeyed` when it genuinely has none. | #3126 |
| The MIMG resolver's class check is equivalent to searching for a resource of an acceptable class | **Falsified — it is strictly weaker, in two ways.** `by_fetch_pc` / `by_srt_offset` / `by_sgpr_base` are first-match-wins **and class-blind**; the resolver took the first hit at a key and nulled it afterwards if the class was wrong. So (a) an image resource sitting behind a buffer-class resource at the same key was unreachable — key collisions are ordinary, not hypothetical: Stray's failing vertex stage carries a `ConstantBuffer` and two `VertexBuffer`s all at `sgpr_base 8` — and (b) a wrong-class hit on the SRT route left `res` non-null, which suppressed the `!res` guard on the SGPR fallback beneath it, so a route that could have resolved was never tried. The two sibling paths already did this correctly (`by_sgpr_base_cls(..., ConstantBuffer)` for `s_buffer_load`, `..., VertexBuffer` for MTBUF); MIMG and MUBUF were the two that did not. Fixed with `image_by_{fetch_pc,srt_offset,sgpr_base}` plus one shared `image_resource_class_satisfies` predicate, so the lookup and the post-check cannot drift apart (the drift that made #2275 worse than no fix). The accepted set is unchanged — verified by enumerating all 256x5 (opcode, class) pairs, 258 accepted, zero disagreements, with a mis-mapped `0x0e` control proving the comparison can detect one — so it can never bind a class the op cannot use. It does change **route precedence**, though, and the first version of this row wrongly said otherwise: where a `fetch_pc` carries a buffer-class entry first and an image-class entry later, the old code fell through to the SRT/SGPR routes and the new code lets the pc route win, so the outcome can be a *different but class-valid* resource rather than only a rejection turned into a resolution. That is the intended contract (exact per-use provenance outranks a table key), not an accident. **It does not by itself resolve any of Stray's five sites**, and that is now measured rather than inferred: `PROSPER_DYNTRACE_FAIL=1` puts all five at `have_t8=0 seed_t8=0` — no traced descriptor and no plausible entry-user-data seed — against 29 `have_t8=1` and 21 `seed_t8=1` elsewhere in the same run, so the discriminator is live and no lookup route could have resolved them. **Do not read that as "the descriptor is absent from memory", and do not attribute it to #305** — both were proposed and withdrawn the same night. `[resdump]` for the vertex site shows the eight dwords present and readable and holding two well-formed V#s (`base=0x302f65b3cc stride=16 num_records=16`, then `base=0x2040f1a5c0 stride=16 num_records=3`); the T# reading of the same bytes is `type=0`, i.e. `bad-image-type`, which is why `seed_t8=0`. And the read is at user-data dword **0** of a declared `[0,20)` window with `[udcand]` reporting `FITS-BLOCK`, so #305's larger-block-than-window condition is not present. `seed_t8=0` is a SHAPE verdict, not a residency one. See `STRAY_STATUS.md` for the three readings that survive and the one `PROSPER_SHARPLOG` field that separates them. | #3126, #1634, `shader_resources.cpp`, `rdna2_emit_alu.cpp` |
| The **user-data block** is a previous pipeline's leftover (from the founding premise, "first draws run with the previous pipeline's PGM + user data" — the *user-data* half; current q1/q3 provenance is summarized under **Current frontier** below) | **Falsified.** This was #305's founding premise and its original title; the issue was **retitled on 2026-08-01** so nobody starts from it. `PROSPER_UDPROV=1` records each SH register's last-write `command_order` and path and carries it into the per-draw snapshot: at **every** failing draw the dwords the shader dereferences were written by the **immediately preceding bind**, a handful of packets before the draw. Identical across 21 measured stages. | #305 |
| The **shader-header registry lookup is stale** (a recycled code allocation resolves an old layout) | **Falsified.** Real hazard in shape — `prosper_agc_shader_header_for_code` returns the *first* match in an append-only registry — but measured `registrations=1` for every failing address across a 2,725-entry registry. | #305 |
| A **bind packet is missing, dropped or mis-ordered** in the decoded stream | **Falsified.** `PROSPER_BINDTRACE=1` logs every Sh `Set*RegsIndirect` packet carrying a program register, in stream order, interleaved with draws: **0 of 141** register arrays apply more than one distinct `(es_lo, rsrc2)` over 193,397 packets, and **300,404 of 300,404** draws fold with the immediately preceding bind. Zero `SetRegsIndirect array unmapped`, zero out-of-range Sh writes. *(Use these corrected figures: the numbers first posted — 434,239 packets, 871,648 of 876,217 — were contaminated by compute dispatches mislabelled `DRAW` by the instrument itself, and the issue body still quotes them. The correction tightens the conclusion.)* | #305 correction comment |
| The stage's user data is the **tail** of the programmed block (a constant seed shift) | **Falsified despite a 9-of-9 numeric fit.** Declared descriptors do land on clean guest pointers exactly `programmed − user_data_range_end` dwords above `USER_DATA_GS_0` — but `USER_SGPR == user_data_range_end` for every stage measured (12/12, 8/8, 24/24, 20/20, 30/30, 32/32), so the hardware loads only that many registers and the stage physically **cannot see** anything above them. A live A/B with the shifted seed raised `exec-recompile-reject` from a 118–141 baseline to **521**. Retained, **off**, as `PROSPER_UD_TAIL_ALIGN`; it must stay off. | #305 |
| **#140 — TYPE-0 AGC data packets** carry the missing bind | **Falsified.** One of #305's two original candidates. The register writes that matter all arrive as ordinary decoded `SET_SH_REG` / `Set*RegsIndirect` packets whose provenance is now directly observable, and none is missing. #140 is closed and unrelated to this path. | #305 |
| **Cross-submit register inheritance** is itself the bug | **Falsified.** Opening a fold with a draw and inheriting the previous submit's bind is the *normal* pattern in this title — 10,806 `q1` folds and 653 `q3` folds do it, against 6,342 and 179 that open with a bind — and is correct on a shared ring. The defect is that the *inherited state* is wrong, not that inheritance happens. | #305 |
| The `atomic_image` ternary in `execute_item` selects between two byte-identical `resource_bytes_for` calls because the **atomic arm lost its `atomic_slice_bytes` extent** | **Falsified by the line's own history.** `git log -L` on both sites (the upload source and the write-back destination) shows the ternary introduced by "gpu: execute padded tiled image atomics" with genuinely different arms — bounded `resource_bytes_for(r, guest_bytes)` for atomic images against the **unbounded** `resource_bytes(r)` otherwise — and then "gpu: honor scalar buffer descriptor bounds" widening the **non-atomic** arm to the bounded call. The arms converged from the other direction; the atomic arm has never been edited, so no extent was dropped. The condition is dead in the strong sense — the two arms are the same expression in the same scope over a pure captureless lambda, so no reachable state can separate them — because `guest_bytes` is reassigned to the PHYSICAL `atomic_slice_bytes * atomic_layers` before the ternary runs, making the atomic-image special case redundant at every later guest-side site. Both ternaries are collapsed; the invariant that made them redundant is now `atomic_image_staging_extents` with a test that reddens if the logical extent is substituted. | #3195, #2265, `gpu/resources/atomic_image_staging.hpp` |
| **#1226's Acb register-file split** is dropping graphics user-data writes | **Falsified.** Measured 1,747 Acb folds with SH offsets confined to `[0x207,0x249]` and **zero** graphics user-data writes, so the split discards nothing hardware would have applied. Candidate closed. | #305 |
| A selected descriptor table's `resolved=0` declines mean the table's **records were unreadable** — Linux's `guest_readable` not committing sparse dmem | **Falsified.** The records are readable; they simply hold data that is not a descriptor. An unselected arena slot carries stale bytes (`base=0`, `size_bytes` read as `0x3f800000`, i.e. `1.0f`), so `decode_buffer_descriptor` rejects it and the whole table was declined. Nothing about residency or sparse commit is involved, and no `guest_readable` change was needed — the fix binds a null descriptor for such a slot while keeping the array's arity exact (#2534). **A residency hypothesis that would have required kernel-side work was really a data-shape question answerable from four dwords.** | #2481, #2534 |
| A **wrong seeding origin** or **header decode** explains the unmapped pointers | **Falsified**, and the same measurement *confirms* two assumptions previously taken on faith: the seeding base `USER_DATA_<stage>_0 + user_data_range_start` (`range_start` is 0 for every shader, `range_end` = last declared direct offset + 2), and the merged-stage convention that user data begins at shader SGPR `s8`. The shader's own `SBASE` equals the header's declared offset in all ten stages disassembled. | #1607, #305 |

**Third-title reproduction, and one counter-example to the paragraph below (2026-09-04, Stray
`PPSA02101`, #3126).** `PROSPER_UDPROV` on the title-screen route reproduces the frontier signature
exactly — pipeline bound at `i564796/q1,f2989`, part of the user-data block rewritten at
`d564819..564823/q3,f2990`, draw at `564827` — and adds two things. **(1) The overwrite is partial and
lands MID-DESCRIPTOR.** The pixel stage's block (`base=0xc`) splits at dword 4 inside an eight-dword
declared image slot: the head is now a V# (`base=0x20e0e42400 stride=16 num_records=8`) while the tail
is still the original T#'s tail, which is exactly why that slot prints `claimed by the V# path`. A
producer rebinding an image slot writes all eight dwords; writing the first four is a different
producer with a four-dword slot at that offset. **(2) The generalisation drawn from the
window survey below does not hold here.** Be precise about which clause falls: the survey's own
sentences are about **vertex** windows ("appears in 8/12/28-dword vertex windows and not in 30/32-dword
vertex windows") plus an observation that every pixel stage in that run had a 30-dword window — those
remain statements about that run. What falls is the conclusion drawn from them, *"window fit, not stage
identity, predicts the result"*: this **pixel** stage declares `[0,30)` and shows the split, so a
30-dword window does not predict its absence. The consequence is measurable rather than theoretical: the
stage declares image slots at dwords 0, 8 and 16 and samples all three, exactly the slots that are
mangled are dropped, and exactly the ops sampling those slots fail — 1:1, with the op sampling the
intact slot never failing. Which slot is intact varies per draw. Details in `STRAY_STATUS.md`
§ *The unresolved image ops*.

**Current frontier.** The source of each write is already observable: for every sampled misfit the
pipeline was bound in a `q1` (Dcb) fold *N*, while the larger user-data block was written in the
following `q3` (`w1KFAHVqpaU` / DcbFinal) fold *N+1*. `PROSPER_UDPROV` retains order, direct/indirect
path, submit origin, fold and Jump depth in the draw snapshot, and `PROSPER_SUBMITORDER` independently
records call order and thread identity before the submit mutex. Submit identity is therefore not a
missing instrument.

That provenance does **not** establish cross-queue ordering as the root cause. Giving DcbFinal a
separate `GpuState` was tested as a default-off A/B: reject counts stayed within route variance and
the 3D world remained identically black, so that split is ruled out as a fix. The earlier GS-versus-PS
acceptance test is retired too. The signature appears in 8/12/28-dword vertex windows and not in
30/32-dword vertex windows; every pixel stage with a declared pointer in the measured run also had a
30-dword window. Window fit, not stage identity, predicts the result. What remains open is the
hardware ordering/ring contract that prevents a larger required block from being paired with the
smaller bound pipeline window.

`test_command_provenance` pins the diagnostic contract synthetically across q1/q2/q3 origins,
top-level folds, an inner Jump, direct/indirect paths and retained write/draw order. It protects the
instrument; it does not fix #305 or assert the still-unknown hardware contract.

**Instruments** (`PROSPER_UDPROV`, `PROSPER_BINDTRACE`, `[udcand]`, `PROSPER_SHADER_HEADER_NEWEST`,
`PROSPER_UD_TAIL_ALIGN`) are on PR #1639 — reuse them rather than rebuilding this measurement.

`PROSPER_SHARPLOG=1` is the fourth one and answers a different question from the rest. They all
describe what the front half *produced*; this one prints what the shader **declared** — the stage's
raw sharp table (per-category counts, each slot's `bits`/`offset_dw`/`size`/EUD residency, and the
direct slots) — and then names the exact reason a declared texture slot was dropped. Between the two
sits a chain of about ten `continue`s, and without the input side "the front half never saw this
resource" and "it saw it and rejected it" produce identical evidence while pointing at different
files. Reach for it before concluding anything about descriptor *recovery*: on Earthion (#1590) it
overturned exactly that conclusion in one run.

**A DCC-compressed SAMPLED surface's cache identity does NOT include its metadata plane, and folding
that plane into the key is both unnecessary and — as first attempted — ineffective.** #3150 /
#3149, 2026-08-31. The persistent compute image cache excluded compressed sampled surfaces on the
theory that base bytes alone could not determine their content; on Stray that exclusion re-detiled
one 4K FP16 surface 3,860 times in 110 s (98.9% of all tiling work, `PROSPER_TILECENSUS`). The
theory is wrong for the *sampled* path: the DCC plane's only consumer there is
`compute_sampled_dcc_fast_clear_rgba8`, whose materialization the admission gate already excludes
via `!sampled_dcc_fast_clear`, so a cacheable entry's upload is detile + unpack of the BASE bytes
alone — which the existing validation already covers. More sharply, `gfx10_dcc_fast_clear_rgba8` is
the **only** DCC decode function in the tree: prosper has no DCC decompressor, so
metadata-independence is a property of the codebase rather than of one call graph. **Do not re-derive
the metadata-in-the-key design**: the first revision of #3150 did, and it was rejected twice over —
the guard was conjoined into only one of three skip proofs (`watch_unchanged` and `exact_unchanged`
both examine the base range alone, so the upload was skipped anyway), and the premise was unnecessary
regardless. The *storage* gate is different and still requires an all-`0xff` plane: a storage
target's base bytes are not authoritative while a live plane exists. `PROSPER_NO_DCC_IMAGE_CACHE=1`
restores the old exclusion for bisection.
