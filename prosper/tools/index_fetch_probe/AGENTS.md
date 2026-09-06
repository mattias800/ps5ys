# AGENTS.md — prosper/tools/index_fetch_probe

Independent bare-Vulkan oracle for **#2961**: does an indexed draw hand the
vertex shader the fetched index-buffer value (correct) or the sequential
vertex ordinal (the falsified #2961 hypothesis — the signal came from
vkprobe's fixed readback window, not from the driver)? The probe exists so
that any future suspicion costs one command instead of a re-derivation. It
shares **no code** with vkprobe — the point of the experiment is answering
whether a divergence lives in the driver or in vkprobe's harness, so
importing vkprobe's pipeline setup would defeat it.

- `probe.c` — the whole program. Vulkan 1.0, no extensions, no layers, one
  64×64 offscreen render pass. A POINTS draw records `(gl_VertexIndex,
  fetched attribute)` per vertex into an SSBO keyed by the fetched value; the
  vertex buffer is identity, so a correct driver writes every indexed slot as
  its own index. Exit 0 fetched / 1 divergence / 2 tool error. The identity
  control list runs first: both hypotheses agree on it, so it proves the
  harness drew without discriminating.
- `shaders/*.spvasm` — readable SPIR-V assembly (assemble with `spirv-as
  --target-env vulkan1.0`); `probe_vert_spv.h` / `probe_frag_spv.h` are the
  generated word lists the binary embeds, so building needs no shader
  compiler. Regenerate the headers after touching the asm.

Results and the run environment are recorded on #2961; update that issue when
a new driver generation is tested.

The standalone `tools/doctor` build can compile the same source as `renderdoc_control`
with `PROSPER_RENDERDOC_CONTROL`: an optional API capture bracket, not a change to the
default probe. Its replay oracle is five draws and the exact final SSBO, including untouched
sentinels. The DONT_CARE color attachment is deliberately **not** a pixel oracle.
See [the capture/replay workflow](../../docs/DEBUGGING_WORKFLOWS.md#renderdoc-prove-capture-replay-and-data-inspection).
