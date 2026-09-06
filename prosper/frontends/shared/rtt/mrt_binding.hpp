#pragma once
#include <atomic>
#include <cstdint>

#include "gpu/execute/gpu_execute.hpp"
#include "gpu/state/render_state.hpp"
#include "shared/rtt/mrt_extent.hpp"
#include "shared/rtt/rtt_scale.hpp"

// THE definition of an active colour binding.
//
// Two things ask it and they must not disagree: pass grouping, which decides how many attachments a
// render pass has and which surfaces they are, and same-pass feedback detection, which decides
// whether a sampled surface is one this pass is writing. A second, looser copy of this rule
// classified stale named state as a live binding, which denies the authoritative direct-GPU path and
// — when no CPU snapshot exists — degrades to guest bytes rather than to a slower correct source.
//
// One asymmetry this header does NOT resolve: for SLOTS 0 AND 1 its readers consult different
// representations. Pass identity is named-first (`mrt_pass_color_binding`), because that is where
// live_renderer takes the address it renders to; the active-binding rule that feeds grouping's
// attachment count and feedback detection is array-first (`mrt_color_binding`). They agree because
// every producer keeps `color_targets[0]`/`[1]` a mirror of the named triple — or leaves it absent —
// not because either rule enforces it.
//
// `mrt_color_alias_mirrored` is that convention written down, and `mrt_check_color_alias_mirror` is
// how a consumer checks it AT THE POINT OF USE. Deliberately not at the producers: an assertion
// beside `realize_draw_item`'s mirror would sit two lines below the assignment it verifies and
// could never fail, which is worse than no check because it reads as coverage (#3026). Which
// representation should win when they disagree is UNDECIDED — no producer emits a divergence, so
// there is no evidence to decide it, and deciding would change the surface the renderer renders to.
// The check therefore reports and changes nothing.
//
// The rule: a slot is active when it has a base, a non-zero write mask, and a format the backend
// accepts. That last term is TOTAL in the current backend -- `backend_color_format` maps every
// unrecognised value, zero included, onto R8G8B8A8_UNORM, so it never rejects anything and a zero
// format means "use the fallback" rather than "undefined". It is kept as a parameter because it is
// the backend's decision to make, not this header's, and a backend that ever narrows it must narrow
// grouping and feedback together. For the ACTIVE-BINDING rule above, the
// named `color0_*` / `color1_*` fields are consulted ONLY when the array representation is absent,
// because `DrawItem` predates the complete array and capture versions through v33 carry the first
// two attachments in those fields. Falling back whenever the array mask merely reads zero is a
// different and wrong rule: a genuinely masked-off slot then inherits a stale named mask.
//
// That "ONLY" scopes to the active-binding rule and does not extend to slot 0's PASS IDENTITY,
// which is named-first for the reason given at `mrt_pass_color_binding`.
namespace prosper::frontend {

// The named `color0_*` / `color1_*` triple a slot aliases. `DrawItem` predates the complete array
// and only ever grew compatibility fields for the first two attachments, so slots 2 and up have no
// named alias and answer with an empty binding; every caller in this header guards that.
inline prosper::gpu::DrawItem::ColorTargetBinding mrt_named_color_alias(
    const prosper::gpu::DrawItem& draw, uint32_t slot) {
    if (slot == 0)
        return prosper::gpu::DrawItem::ColorTargetBinding{
            draw.color0_base, draw.color0_width, draw.color0_height};
    if (slot == 1)
        return prosper::gpu::DrawItem::ColorTargetBinding{
            draw.color1_base, draw.color1_width, draw.color1_height};
    return prosper::gpu::DrawItem::ColorTargetBinding{};
}

inline prosper::gpu::DrawItem::ColorTargetBinding mrt_color_binding(
    const prosper::gpu::DrawItem& draw, uint32_t slot) {
    auto binding = draw.color_targets[slot];
    if (!binding.base && !binding.width && !binding.height && slot < 2u)
        binding = mrt_named_color_alias(draw, slot);
    return binding;
}

// THE surface a colour pass renders to, for one slot -- pass IDENTITY, as distinct from the
// active-binding rule above.
//
// Slot 0 is taken from the NAMED triple whenever it names a surface, because that is the surface
// the pass actually renders to: live_renderer takes `pass_bases[0]` from `color0_base` and the
// framebuffer extent from `color0_width`/`color0_height`, while every slot above 0 goes through the
// array. Reading slot 0's identity from the array instead makes GROUPING and TARGET SELECTION
// consult different representations of one attachment -- and two draws that render to DIFFERENT
// addresses then share a pass, so the second draw's target is silently discarded and its pixels are
// never published (#3023).
//
// On captured and live draws the two representations are identical, so this costs nothing there.
// For captures a divergence is not merely absent but INEXPRESSIBLE: the wire format carries slots 2
// and up only, and `restore_legacy_color_target_aliases` re-derives slots 0/1 from the named triple
// on every load. For live draws the same mirror sits at the single success exit of
// `realize_draw_item`. The two differ only for a caller that builds a DrawItem directly and updates
// the named aliases alone: exactly the shape that mirror exists to repair, and the shape the render
// fixtures construct. Falling back to `mrt_color_binding` when `color0_base` is 0 keeps the previous
// answer for a draw that names no slot-0 surface at all.
inline prosper::gpu::DrawItem::ColorTargetBinding mrt_pass_color_binding(
    const prosper::gpu::DrawItem& draw, uint32_t slot) {
    if (slot == 0 && draw.color0_base) return mrt_named_color_alias(draw, 0);
    return mrt_color_binding(draw, slot);
}

// Does an aliased slot's ARRAY entry still mirror its named triple?
//
// THE CONVENTION, written down. For slots 0 and 1 `DrawItem::color_targets[slot]` must either equal
// the named `colorN_*` triple or be ABSENT (all three fields zero, the legacy shape the fallback in
// `mrt_color_binding` exists for). Every producer honours it -- `realize_draw_item`'s success exit
// and its failure record, the three DrawItem<->capture conversions, and
// `restore_legacy_color_target_aliases` on every capture load -- and until #3026 nothing checked it.
//
// Given the convention, all four slot-0 readers return the same triple: this header's
// `mrt_color_binding` and `mrt_pass_color_binding`, live_renderer's `pass_bases[0]`, and
// live_renderer's framebuffer extent. Break it and they answer three different ways, which is the
// class #3023 was one instance of. Slots 2 and up have no named alias, so nothing there can diverge.
inline bool mrt_color_alias_mirrored(const prosper::gpu::DrawItem& draw, uint32_t slot) {
    if (slot > 1u) return true;
    const auto& carried = draw.color_targets[slot];
    if (!carried.base && !carried.width && !carried.height) return true;   // absent, not divergent
    const auto named = mrt_named_color_alias(draw, slot);
    return carried.base == named.base && carried.width == named.width &&
           carried.height == named.height;
}

// How many aliased slots this process has observed diverging. Exposed so a test can assert the
// CONSUMER saw a hand-built divergence, not merely that the predicate above can compute one -- the
// distinction this header already records under `mrt_direct_serves`, where tests that exercised the
// helper instead of the decision stayed green while production was reverted.
inline std::atomic<uint64_t>& mrt_color_alias_divergence_counter() {
    static std::atomic<uint64_t> count{0};
    return count;
}

inline uint64_t mrt_color_alias_divergences() {
    return mrt_color_alias_divergence_counter().load(std::memory_order_relaxed);
}

// Check both aliased slots of one draw where a consumer is about to act on them. `report` receives
// the slot, the carried array entry, the named alias, and this divergence's process-wide ordinal;
// the caller owns it, so this header stays free of I/O and of any rate-limit policy. Returns how
// many slots diverged -- 0 for every draw any production producer emits.
template <typename Report>
uint32_t mrt_check_color_alias_mirror(const prosper::gpu::DrawItem& draw, Report report) {
    uint32_t diverged = 0;
    for (uint32_t slot = 0; slot < 2u; ++slot) {
        if (mrt_color_alias_mirrored(draw, slot)) continue;
        ++diverged;
        report(slot, draw.color_targets[slot], mrt_named_color_alias(draw, slot),
               mrt_color_alias_divergence_counter().fetch_add(1, std::memory_order_relaxed) + 1u);
    }
    return diverged;
}

// Raw guest format for a slot, with the same named-field fallback. Left as the raw value so callers
// that need a backend VkFormat can map it themselves without this header depending on the backend.
inline uint32_t mrt_raw_format(const prosper::gpu::DrawItem& draw, uint32_t slot) {
    uint32_t raw = draw.ps.color_targets[slot].format;
    if (slot == 0 && !raw) raw = draw.ps.color0_format;
    if (slot == 1 && !raw) raw = draw.ps.color1_format;
    return raw;
}

// The write mask that governs this slot, with the named fallback taken only when the array
// representation is ABSENT -- matching pass grouping exactly.
inline uint32_t mrt_write_mask(const prosper::gpu::DrawItem& draw, uint32_t slot) {
    const auto& target = draw.ps.color_targets[slot];
    if (slot == 0 && !target.format && !draw.color_targets[slot].base)
        return draw.ps.color_write_mask;
    if (slot == 1 && !target.format && !draw.color_targets[slot].base)
        return draw.ps.color1_write_mask;
    return target.write_mask;
}

// The base this slot is actively writing, or 0. `format_defined` decides whether a raw guest format
// counts as accepted; callers pass their own mapping so this header stays backend-free. Note it is
// total in today's backend -- see the file comment; do not read this parameter as evidence that a
// zero format is rejected anywhere.
template <typename FormatDefined>
uint64_t mrt_active_color(const prosper::gpu::DrawItem& draw, uint32_t slot,
                          FormatDefined format_defined) {
    const auto binding = mrt_color_binding(draw, slot);
    if (!binding.base) return 0;
    if (!mrt_write_mask(draw, slot)) return 0;
    if (!format_defined(mrt_raw_format(draw, slot))) return 0;
    return binding.base;
}

// The active attachment prefix: one past the highest active slot, never less than 1.
template <typename FormatDefined>
uint32_t mrt_active_color_count(const prosper::gpu::DrawItem& draw, FormatDefined format_defined) {
    uint32_t count = 1;
    for (uint32_t slot = 1; slot < prosper::gpu::kColorTargetCount; ++slot)
        if (mrt_active_color(draw, slot, format_defined)) count = slot + 1;
    return count;
}

// Can two consecutive draws share one backend colour pass?
//
// A target address is not a complete attachment identity.  Packed mip tails legitimately give two
// rendered levels the same guest address while their extents still differ.  Grouping only on
// address and format made GTA V's 64x32 and 32x16 R11G11B10F levels share
// one 64x32 Vulkan attachment.  The missing last level was then sampled as undefined max-float data
// by deferred lighting.  Unknown 0x0 extents remain non-conflicting, matching mrt_extent_conflicts.
template <typename FormatDefined, typename FormatAt>
bool mrt_same_color_pass(const prosper::gpu::DrawItem& first,
                         const prosper::gpu::DrawItem& candidate,
                         FormatDefined format_defined, FormatAt format_at) {
    // Slot 0's identity is the surface the pass RENDERS to, which is `mrt_pass_color_binding` --
    // named-first -- and not the active-binding rule. That distinction is the whole of #3023; the
    // reasoning, and why it costs nothing on captured or live draws, is on that function.
    const uint32_t count = mrt_active_color_count(first, format_defined);
    if (mrt_active_color_count(candidate, format_defined) != count) return false;
    for (uint32_t slot = 0; slot < count; ++slot) {
        const auto a = mrt_pass_color_binding(first, slot);
        const auto b = mrt_pass_color_binding(candidate, slot);
        const uint64_t a_base = slot == 0 ? a.base
            : mrt_active_color(first, slot, format_defined);
        const uint64_t b_base = slot == 0 ? b.base
            : mrt_active_color(candidate, slot, format_defined);
        if (a_base != b_base || format_at(first, slot) != format_at(candidate, slot))
            return false;
        const bool active = slot == 0 || a_base || b_base;
        if (active && mrt_extent_conflicts(a.width, a.height, b.width, b.height))
            return false;
    }
    return true;
}

// Can two consecutive draws share one pass given their FIXED-FUNCTION RESOLVE identity?
//
// A CB_COLOR_CONTROL.MODE=RESOLVE draw is not an ordinary draw: live_renderer answers it with a
// straight copy of the already-rendered `color0_base` surface into `color1_base` and then ends the
// group, because a resolve is a copy and not a render. Two consequences follow, and the predicate
// has to carry both.
//
// 1. A resolve must never group with an ordinary draw. It shares `color0_base` with the scene it
//    resolves and reports `mrt_active_color(draw, 1) == 0` -- a fixed-function resolve exports
//    nothing, so its `color1_write_mask` is 0 and slot 1 contributes no attachment -- so on colour
//    identity alone it would merge with those scene draws, and the copy's `continue` would then
//    silently drop every ordinary draw grouped after it.
//
// 2. Two resolves must not group unless they resolve into the SAME destination. `cb_resolve` alone
//    is a boolean and cannot separate one resolve from another, while slot 1 -- the slot the
//    destination lives in -- is inactive for every resolve and so contributes no discriminating
//    term to `mrt_same_color_pass`. Consecutive resolves sharing a source and naming DIFFERENT
//    `color1_base` destinations therefore satisfied every grouping term, became one pass, and had
//    one copy performed from the group's first draw; every other destination was discarded with no
//    copy, no render and no diagnostic (#3025). The destination the copy acts with is the raw
//    `color1_base`, so that is the field the group must be keyed on -- the same principle #3023
//    established for slot 0: THE IDENTITY A PASS GROUPS ON MUST BE THE IDENTITY IT ACTS WITH.
//
// With the destination in the key, every member of a resolve group names one source and one
// destination, so the single copy the group performs is exactly right for all of them and no member
// needs its own.
//
// Deliberately keyed on the RAW named field rather than on `mrt_active_color(draw, 1, ...)`, for
// the same reason the copy reads it raw: a fixed-function resolve's slot 1 is never active, so the
// active-binding rule reports 0 for every resolve and would key the group on a constant.
inline bool mrt_same_resolve_pass(const prosper::gpu::DrawItem& first,
                                  const prosper::gpu::DrawItem& candidate) {
    if (first.ps.cb_resolve != candidate.ps.cb_resolve) return false;
    if (!first.ps.cb_resolve) return true;
    return first.color1_base == candidate.color1_base;
}

// Does this draw bind the sampled VIEW as any ACTIVE colour target? Address alone is insufficient:
// packed mip tails may give two levels the same guest base even though they are separate Vulkan
// images. Known, conflicting extents therefore prove that the sampled view is not the attachment.
// Unknown extents remain conservative and count an address match as feedback -- unlike pass
// grouping, this decision protects us from binding one image simultaneously for sampling and
// rendering, so absence of evidence cannot make the operation safe.
template <typename FormatDefined>
bool mrt_draw_binds_target_view(const prosper::gpu::DrawItem& draw, uint64_t addr,
                                uint32_t sampled_width, uint32_t sampled_height,
                                FormatDefined format_defined) {
    if (!addr) return false;
    for (uint32_t slot = 0; slot < prosper::gpu::kColorTargetCount; ++slot) {
        if (mrt_active_color(draw, slot, format_defined) != addr) continue;
        const auto binding = mrt_color_binding(draw, slot);
        if (mrt_extent_conflicts(binding.width, binding.height,
                                 sampled_width, sampled_height))
            continue;
        return true;
    }
    return false;
}

// Address-only compatibility wrapper. With an unknown sampled extent it deliberately keeps the
// historical conservative behaviour.
template <typename FormatDefined>
bool mrt_draw_binds_target(const prosper::gpu::DrawItem& draw, uint64_t addr,
                           FormatDefined format_defined) {
    return mrt_draw_binds_target_view(draw, addr, 0u, 0u, format_defined);
}

// The two materialization decisions that key on feedback, as seams that OWN their gate.
//
// They exist because the gate is the interesting part and a helper called beside it is not: with the
// comparison written inline at each call site, reverting one to `sampled != draw.color0_base` left
// every test green, since the tests exercised the helper rather than the decision.
//
// The two are coupled, which is why they must agree. `mrt_direct_serves` deciding TRUE suppresses
// the lazy CPU materialisation for that resource; if the later bind then refuses the direct image --
// which it must, when the sample really is one of this pass's targets -- there is no snapshot left
// and the resource degrades to guest bytes. So a feedback collision has to be seen by BOTH, and it
// used to be seen by neither above slot 1.

// May the retained GPU image serve this sample without materializing guest/CPU bytes? An ordinary
// non-feedback sample borrows the target directly. When `feedback_copy_supported` is true, an exact
// attachment collision is also admitted because the backend snapshots the prior attachment version
// into a distinct sampled image before beginning the render pass. The default preserves the
// conservative contract for callers that do not implement that GPU copy.
template <typename FormatDefined>
bool mrt_direct_serves(const prosper::gpu::DrawItem& draw, uint64_t sampled,
                       uint32_t sampled_width, uint32_t sampled_height,
                       bool is_storage_image, uint32_t img_dim, uint32_t layers,
                       uint32_t samples, bool extent_compatible,
                       bool has_persistent_target, FormatDefined format_defined,
                       bool feedback_copy_supported = false) {
    const bool feedback = mrt_draw_binds_target_view(
        draw, sampled, sampled_width, sampled_height, format_defined);
    return !is_storage_image && rtt_single_layer_sample_shape(img_dim, layers, samples) &&
           extent_compatible && has_persistent_target &&
           (!feedback || feedback_copy_supported);
}

// May the uniform-colour fast path serve this sample? `preconditions` folds the caller's own
// non-feedback terms (not a storage image, a plain 2D view, not in a mip tail, a uniform cache entry
// with a usable extent) so this seam owns exactly the feedback gate and nothing it cannot see.
template <typename FormatDefined>
bool mrt_uniform_live_serves(const prosper::gpu::DrawItem& draw, uint64_t sampled,
                             uint32_t sampled_width, uint32_t sampled_height,
                             bool preconditions, FormatDefined format_defined) {
    return preconditions &&
           !mrt_draw_binds_target_view(draw, sampled, sampled_width, sampled_height,
                                       format_defined);
}

}  // namespace prosper::frontend
