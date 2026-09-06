# Vulkan runtime and modernization (#3414)

The live graphics backend, standalone live compute backend, and app's fallback presentation device
require Vulkan **1.4**. Each instance-creation path resolves `vkEnumerateInstanceVersion` before
requesting 1.4; an absent function, failed query, or older loader produces a diagnostic. Physical
device selection independently rejects older devices. Standalone compatibility probes and synthetic
Vulkan harnesses can still request older versions; they do not define the shipping runtime's floor.

Build these backends with Vulkan 1.4 headers. Linux is the primary target; the contract uses standard
Vulkan APIs and does not depend on an AMD extension or an experimental driver flag. NVIDIA and
Windows devices must satisfy the same version and feature checks. This change was tested on Linux
RADV, not certified on NVIDIA or Windows. MoltenVK devices exposing less than 1.4 are no longer
accepted, as permitted by the owner's platform-priority decision in #3414.

## Features already consumed

Graphics now queries descriptor indexing, buffer int64 atomics, robust image access, and subgroup
size control as core features. Standalone compute queries its descriptor indexing and int64 atomics
the same way. An absent promoted extension name no longer disables a supported core feature.
`VkPhysicalDeviceFeatures2` still checks the individual bits, and device creation requests only the
bits in use. Shared compute inherits the renderer's **enabled** capabilities.

Host image copy support is logged for the selected physical device, explicitly distinguished from
enablement. It is not enabled or used by this change. Timeline semaphores, synchronization2, dynamic
rendering, and push descriptors likewise remain follow-up work, not benefits delivered by changing
the API version.

Validation: the combined change passes 11 focused CTest cases covering audio, retained targets,
shader interfaces, SPIR-V validation, device publication, and live compute. With validation enabled,
`game_compute_exec` still emits the existing #1710 push-constant IDs `07987`/`08602`, and
`interp_render` emits the existing #1715 geometry-invocation ID `00715`; all three are recorded in
`tools/vkval/allowlist.txt`. This is not a claim of a validation-clean whole suite. A separate
45-second windowed Sonic run presented 452 frames with an active validation messenger and **zero**
validation warnings/errors, without experimental driver flags.

## Host image copy: checked capability, not assumed hardware limitation

On the measured Radeon 8060S / RADV STRIX_HALO device with Mesa 26.1.4, `hostImageCopy=false` by
default. The concurrently enumerated llvmpipe device reports true; its result must not be attributed
to the hardware GPU. Vulkan 1.4 explicitly permits omitting this feature when the implementation
provides an additional transfer-capable queue.

This is a driver default, not proof the chip cannot implement the operation. Mesa 26.1.4 gates it
behind `RADV_EXPERIMENTAL=hic`; a capability-only probe with that process-local switch reported true
on the same Radeon. Mesa 26.2 enables it by default on GFX10.3 and newer. No system settings or driver
packages were changed, and the experiment is not part of the runtime launch contract.

References: [Vulkan host image copy](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_host_image_copy.html),
[Mesa 26.1.4 feature gate](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.1.4/src/amd/vulkan/radv_physical_device.c),
[Mesa 26.2 feature gate](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.2.0/src/amd/vulkan/radv_physical_device.c).

A future optional implementation must check the feature, image usage/format support, permitted host
copy layouts, and device-access performance properties, with the existing staging path retained.
GPU writes still need a host-read availability dependency and completion before a host copy reads
them. The API removes staging machinery; it does not remove required synchronization or format
conversion. Test it on drivers that advertise support normally, without requiring experimental flags.

## Measurements after the retained-target fix

The #3415 change eliminated the observed same-batch CPU readbacks in two comparable 105-second
Sonic Frontiers intro runs and raised presented frames from 470 to 1,086. See
[title evidence](SONIC_FRONTIERS_STATUS.md). That improvement precedes the Vulkan version change.

A subsequent 75-second windowed CPU profile at `d32e4cee` recorded 2,691 `cpu-clock:u` samples at
99 Hz, with no lost samples. It included one scheduled screenshot and is a cost profile, not a
clean comparative FPS benchmark. Self CPU sample shares included memset 18.80%, memmove 17.17%,
compute texel conversion 11.78%, tiling 7.25%, memcmp 5.72%, and detiling 2.64%. Guest native code and
blocked time are not fully represented by these host userspace samples.

The renderer's cumulative timings reported about 0.77 ms waiting per backend submission versus
0.01 ms for pipeline creation; compute dispatches averaged 3.13 ms including their surrounding work.
The compute backend already reuses its dispatch fence. Replacing that fence with a timeline semaphore
alone would not remove a wait where the CPU immediately consumes the dispatch result. Pipeline
permutation reduction is also not established as the next bottleneck by this run.

Prioritize the measured image conversion/copy traffic and dependencies forcing synchronous result
consumption. Adopt synchronization2 with producer/consumer stage knowledge, not by mechanically
renaming `ALL_COMMANDS` barriers. Dynamic rendering can simplify object lifetime, but neither it nor
the version bump has a measured performance gain here yet. Audio is recognizable after #3411 and
its production rate improved with #3415; sustained delivery remains below the required sample rate.
