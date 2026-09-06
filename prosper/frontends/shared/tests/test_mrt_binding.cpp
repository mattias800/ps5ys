// One active-colour-binding policy, exercised through a real DrawItem.
//
// Pass grouping and same-pass feedback detection must agree about what an active binding is. They
// did not: the feedback path carried a second, looser copy that fell back to the named slot-0/1
// write mask whenever the ARRAY mask read zero, and never required a defined format. That
// classifies stale named state as a live binding, which denies the authoritative direct-GPU path --
// and where no CPU snapshot exists, degrades to guest bytes rather than to a slower correct source.
#include <cstdio>
#include <cstdint>

#include "shared/rtt/mrt_binding.hpp"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

namespace {

// Production's predicate is `backend_color_format(raw) != VK_FORMAT_UNDEFINED`, and
// backend_color_format's default branch maps EVERY unrecognised value -- including zero -- to
// R8G8B8A8_UNORM. So the production predicate is total: it returns true for every input, and the
// format term in the active-binding rule never rejects anything.
//
// That is deliberate, not an oversight. Synthetic callers and legacy captures omit CB_COLOR_INFO and
// rely on the backend's established RGBA8 fallback; requiring a decoded format would discard those
// otherwise-valid draws (the same reasoning the executor's `color_effect` check records). A live
// measurement agrees: PROSPER_MRT_CENSUS reported "format known" for every slot in all 16,384 pass
// groups of a routed GTA V boot -- the column is constant because the predicate is.
//
// Modelled here as the constant it is. An earlier version of this test modelled `raw != 0` and
// asserted a zero format made a slot inactive, which is a behaviour production does not have -- the
// central policy test proving the opposite of the policy. `mrt_format_total_mapping` in
// test_recompiled_fragment pins the backend mapping this constant depends on, so a change there
// fails rather than silently invalidating this file.
bool format_defined(uint32_t raw) { (void)raw; return true; }

uint32_t format_at(const prosper::gpu::DrawItem& draw, uint32_t slot) {
    return prosper::frontend::mrt_raw_format(draw, slot);
}

prosper::gpu::DrawItem make_draw() {
    prosper::gpu::DrawItem draw{};
    return draw;
}

}  // namespace

int main() {
    using prosper::frontend::mrt_active_color;
    using prosper::frontend::mrt_active_color_count;
    using prosper::frontend::mrt_draw_binds_target;
    using prosper::frontend::mrt_draw_binds_target_view;
    using prosper::frontend::mrt_same_color_pass;
    using prosper::frontend::mrt_same_resolve_pass;

    constexpr uint64_t kMrt0 = 0x2050000000ull;
    constexpr uint64_t kMrt2 = 0x2083e00000ull;

    // A slot in the ARRAY representation is active with base + format + mask.
    {
        auto draw = make_draw();
        draw.color_targets[2].base = kMrt2;
        draw.ps.color_targets[2].format = 37u;
        draw.ps.color_targets[2].write_mask = 0xfu;
        CHECK(mrt_active_color(draw, 2, format_defined) == kMrt2);
        CHECK(mrt_draw_binds_target(draw, kMrt2, format_defined));
        CHECK(mrt_active_color_count(draw, format_defined) == 3u);
        // The case the MRT0-only feedback rule missed entirely.
        CHECK(!mrt_draw_binds_target(draw, 0x2099cc0000ull, format_defined));
    }

    // A base with a ZERO write mask is not a target. Stale bases are sticky in the guest's register
    // file, so treating one as a binding both over-counts attachments and manufactures feedback.
    {
        auto draw = make_draw();
        draw.color_targets[2].base = kMrt2;
        draw.ps.color_targets[2].format = 37u;
        draw.ps.color_targets[2].write_mask = 0u;
        CHECK(mrt_active_color(draw, 2, format_defined) == 0u);
        CHECK(!mrt_draw_binds_target(draw, kMrt2, format_defined));
    }

    // A base whose format was never seen IS still active: zero means "use the backend's RGBA8
    // fallback", not "undefined". Asserted explicitly because the opposite reads as the safer
    // expectation and is wrong -- rejecting it would drop attachments for every synthetic caller and
    // every legacy capture that omits CB_COLOR_INFO.
    {
        auto draw = make_draw();
        draw.color_targets[2].base = kMrt2;
        draw.ps.color_targets[2].format = 0u;
        draw.ps.color_targets[2].write_mask = 0xfu;
        CHECK(mrt_active_color(draw, 2, format_defined) == kMrt2);
        CHECK(mrt_draw_binds_target(draw, kMrt2, format_defined));
    }

    // Legacy shape: DrawItem predates the complete array, and captures through v33 carry the first
    // two attachments in the named fields. With the array ABSENT the named state is authoritative.
    {
        auto draw = make_draw();
        draw.color0_base = kMrt0;
        draw.ps.color0_format = 37u;
        draw.ps.color_write_mask = 0xfu;
        CHECK(mrt_active_color(draw, 0, format_defined) == kMrt0);
        CHECK(mrt_draw_binds_target(draw, kMrt0, format_defined));
    }

    // THE defect this policy exists to prevent. The array representation IS present for slot 0 and
    // says the slot is masked off, while stale named state still carries a non-zero mask. The looser
    // rule fell back to the named mask whenever the array mask read zero and called this a binding.
    {
        auto draw = make_draw();
        draw.color_targets[0].base = kMrt0;
        draw.ps.color_targets[0].format = 37u;
        draw.ps.color_targets[0].write_mask = 0u;   // the array says: masked off
        draw.color0_base = kMrt0;
        draw.ps.color0_format = 37u;
        draw.ps.color_write_mask = 0xfu;            // stale named state says otherwise
        CHECK(mrt_active_color(draw, 0, format_defined) == 0u);
        CHECK(!mrt_draw_binds_target(draw, kMrt0, format_defined));
    }

    // A zero address never binds, and a draw with nothing bound binds nothing.
    {
        auto draw = make_draw();
        CHECK(!mrt_draw_binds_target(draw, 0, format_defined));
        CHECK(!mrt_draw_binds_target(draw, kMrt0, format_defined));
        CHECK(mrt_active_color_count(draw, format_defined) == 1u);
    }

    // A packed mip tail may place two different rendered levels at the same guest address.  They
    // must form different backend passes: one Vulkan attachment has one extent and cannot represent
    // both GTA V's 64x32 and 32x16 levels.  Exact peers still group, while an unknown legacy extent
    // remains non-conflicting rather than inventing evidence.
    {
        constexpr uint64_t kTail = 0x209c980000ull;
        auto mip4 = make_draw();
        mip4.color_targets[0] = {kTail, 64u, 32u};
        mip4.ps.color_targets[0].format = 122u;
        mip4.ps.color_targets[0].write_mask = 0xfu;

        auto peer = mip4;
        CHECK(mrt_same_color_pass(mip4, peer, format_defined, format_at));

        auto mip5 = mip4;
        mip5.color_targets[0].width = 32u;
        mip5.color_targets[0].height = 16u;
        CHECK(!mrt_same_color_pass(mip4, mip5, format_defined, format_at));

        // The mip-5 draw samples mip 4 by the same guest address. Different known extents prove
        // that these are different attachment views, so this is not feedback. An exact peer is;
        // an unknown extent remains conservative because binding the wrong live image is unsafe.
        CHECK(!mrt_draw_binds_target_view(mip5, kTail, 64u, 32u, format_defined));
        CHECK(mrt_draw_binds_target_view(mip5, kTail, 32u, 16u, format_defined));
        CHECK(mrt_draw_binds_target_view(mip5, kTail, 0u, 0u, format_defined));

        auto legacy_unknown = mip4;
        legacy_unknown.color_targets[0].width = 0u;
        legacy_unknown.color_targets[0].height = 0u;
        CHECK(mrt_same_color_pass(mip4, legacy_unknown, format_defined, format_at));
    }

    // The two materialization decisions, at the seams that own their gate. Written against the
    // decisions rather than the helper because the previous round's tests called
    // mrt_draw_binds_target() directly and stayed green when either production site was reverted to
    // `sampled != draw.color0_base`.
    {
        using prosper::frontend::mrt_direct_serves;
        using prosper::frontend::mrt_uniform_live_serves;

        auto draw = make_draw();
        draw.color_targets[2].base = kMrt2;
        draw.ps.color_targets[2].format = 37u;
        draw.ps.color_targets[2].write_mask = 0xfu;

        // Sampling an unrelated surface: the direct image serves.
        CHECK(mrt_direct_serves(draw, 0x2099cc0000ull, 64u, 32u,
                                /*is_storage_image=*/false, /*img_dim=*/1u, 1u, 1u,
                                /*extent_compatible=*/true, /*has_persistent_target=*/true,
                                format_defined));
        // Sampling THIS pass's MRT2: it must not, or the descriptor and the colour attachment
        // become the same image.
        CHECK(!mrt_direct_serves(draw, kMrt2, 64u, 32u, false, 1u, 1u, 1u, true, true,
                                 format_defined));
        CHECK(mrt_direct_serves(draw, kMrt2, 64u, 32u, false, 1u, 1u, 1u, true, true,
                                format_defined, /*feedback_copy_supported=*/true));
        // The non-feedback preconditions still gate it.
        CHECK(!mrt_direct_serves(draw, 0x2099cc0000ull, 64u, 32u,
                                 /*is_storage_image=*/true, 1u, 1u, 1u, true, true,
                                 format_defined));
        CHECK(mrt_direct_serves(draw, 0x2099cc0000ull, 64u, 32u,
                                 false, /*img_dim=*/5u, 1u, 1u, true, true,
                                 format_defined));
        CHECK(!mrt_direct_serves(draw, kMrt2, 64u, 32u, false, 5u, 1u, 1u,
                                 true, true, format_defined));
        CHECK(mrt_direct_serves(draw, kMrt2, 64u, 32u, false, 5u, 1u, 1u,
                                true, true, format_defined, true));
        CHECK(!mrt_direct_serves(draw, kMrt2, 64u, 32u, false, 5u, 2u, 1u,
                                 true, true, format_defined, true));
        CHECK(!mrt_direct_serves(draw, kMrt2, 64u, 32u, false, 5u, 1u, 4u,
                                 true, true, format_defined, true));
        CHECK(!mrt_direct_serves(draw, kMrt2, 64u, 32u, false, 2u, 1u, 1u,
                                 true, true, format_defined, true));
        CHECK(!mrt_direct_serves(draw, 0x2099cc0000ull, 64u, 32u,
                                 false, 1u, 1u, 1u, /*extent_compatible=*/false,
                                 true, format_defined));
        CHECK(!mrt_direct_serves(draw, 0x2099cc0000ull, 64u, 32u, false, 1u, 1u, 1u, true,
                                 /*has_persistent_target=*/false, format_defined));

        // The uniform fast path carries the same gate. It and mrt_direct_serves must agree: if this
        // one said yes on a collision the CPU materialisation would be skipped, and the direct bind
        // would then be refused with no snapshot left to fall back to.
        CHECK(mrt_uniform_live_serves(draw, 0x2099cc0000ull, 64u, 32u,
                                      /*preconditions=*/true,
                                      format_defined));
        CHECK(!mrt_uniform_live_serves(draw, kMrt2, 64u, 32u, true, format_defined));
        CHECK(!mrt_uniform_live_serves(draw, 0x2099cc0000ull, 64u, 32u,
                                       /*preconditions=*/false,
                                       format_defined));

        // A slot that is bound but MASKED OFF is not a target, so sampling it is not feedback and
        // the direct path must remain available -- denying it would push a live surface onto a
        // slower path, or onto guest bytes when no snapshot exists.
        auto masked = make_draw();
        masked.color_targets[2].base = kMrt2;
        masked.ps.color_targets[2].format = 37u;
        masked.ps.color_targets[2].write_mask = 0u;
        CHECK(mrt_direct_serves(masked, kMrt2, 64u, 32u, false, 1u, 1u, 1u, true, true,
                                format_defined));
    }

    // Slot 0's pass identity is the surface the pass RENDERS to.
    //
    // live_renderer takes `pass_bases[0]` from the named `color0_base`, while every slot above 0
    // goes through the array. Grouping therefore has to read slot 0 from the same place: when it
    // read the array instead, two draws rendering to DIFFERENT addresses grouped into one pass and
    // the second draw's target was silently discarded -- it produced no pass, published no pixels,
    // and the chain returned the previous draw's surface at the previous draw's extent.
    //
    // Every production producer derives `color_targets[0]` from the named fields rather than
    // filling it independently -- `restore_legacy_color_target_aliases` on every capture load, and
    // the mirror at `realize_draw_item`'s single success exit for live draws -- so the two agree on
    // captured and live draws and this costs nothing there. They diverge only for a caller that
    // builds a DrawItem directly and sets the named aliases alone: the "direct/synthetic callers"
    // that mirror exists to repair, and the shape the render fixtures construct. Regression for
    // four gpu_capture_render_replay arms.
    {
        auto producer = make_draw();
        producer.color_targets[0].base = 0x200000ull;  // a stale mirror of an earlier target
        producer.color0_base = 0x300000ull;
        producer.color0_width = 32u;
        producer.color0_height = 2u;

        auto consumer = producer;                      // same stale array entry ...
        consumer.color0_base = 0x400000ull;            // ... a different rendered surface
        consumer.color0_width = 16u;
        consumer.color0_height = 16u;
        CHECK(!mrt_same_color_pass(producer, consumer, format_defined, format_at));
        CHECK(mrt_same_color_pass(producer, producer, format_defined, format_at));

        // Isolate the BASE term: same named extent, a different rendered surface. Without this arm
        // the set is satisfied by a change that keeps the ARRAY base and takes only the named
        // EXTENT -- which gets this fix's whole subject, the address the pass renders to, backwards.
        // The arm above cannot catch that mutant because it differs in base AND extent, and the one
        // below isolates the extent; nothing else here varies the base alone.
        auto other_surface = producer;
        other_surface.color0_base = 0x400000ull;  // extents stay 32x2
        CHECK(!mrt_same_color_pass(producer, other_surface, format_defined, format_at));

        // The series added the extent term to stop GTA V's 64x32 and 32x16 mip levels sharing one
        // attachment. Reading slot 0 from the array defeated that term as well, because a stale
        // array entry carries no extent and unknown extents never conflict: same address, both
        // extents 0x0, grouped. The named extents are what conflict.
        auto smaller_level = producer;
        smaller_level.color0_width = 16u;
        smaller_level.color0_height = 16u;
        CHECK(!mrt_same_color_pass(producer, smaller_level, format_defined, format_at));

        // With no named slot-0 surface the array remains the answer, exactly as before.
        auto array_only = make_draw();
        array_only.color_targets[0].base = 0x500000ull;
        auto array_other = array_only;
        array_other.color_targets[0].base = 0x600000ull;
        CHECK(!mrt_same_color_pass(array_only, array_other, format_defined, format_at));
        CHECK(mrt_same_color_pass(array_only, array_only, format_defined, format_at));
    }

    // #3026 -- the named/array mirror for slots 0 and 1, and a guard that CAN fail.
    //
    // THE CONVENTION: for the two aliased slots, `DrawItem::color_targets[slot]` either equals the
    // named `colorN_*` triple or is ABSENT (all three fields zero, the legacy shape the fallback in
    // `mrt_color_binding` exists for). Every production producer honours it -- `realize_draw_item`'s
    // success exit and its failure record, the three DrawItem<->capture conversions, and
    // `restore_legacy_color_target_aliases`, which runs on every capture load because the wire
    // format carries slots 2 and up only. Nothing checked it, and #3023 is what happens when a
    // consumer starts depending on which of the two it reads.
    //
    // The obvious guard is void, and these arms deliberately do not build it: an assertion inside
    // `realize_draw_item` would sit two lines below the mirror it verifies, so it would pass forever
    // and catch nothing. The predicate is checked at the CONSUMER instead -- live_renderer's pass
    // loop, where the pass target is chosen -- and every divergent instance below is built BY HAND,
    // because no producer can emit one.
    {
        using prosper::frontend::mrt_check_color_alias_mirror;
        using prosper::frontend::mrt_color_alias_divergences;
        using prosper::frontend::mrt_color_alias_mirrored;
        using prosper::frontend::mrt_color_binding;
        using prosper::frontend::mrt_named_color_alias;
        using prosper::frontend::mrt_pass_color_binding;
        using Binding = prosper::gpu::DrawItem::ColorTargetBinding;

        auto same = [](Binding a, Binding b) {
            return a.base == b.base && a.width == b.width && a.height == b.height;
        };
        // THE DEMONSTRATION behind "nothing changes for any reachable draw". Under the convention
        // every slot-0/1 reader in the renderer returns one triple: grouping's pass identity
        // (`mrt_pass_color_binding`), the active-binding rule behind the attachment count and
        // feedback detection (`mrt_color_binding`), and live_renderer's own `pass_bases[0]` plus its
        // framebuffer extent -- which are the raw named fields, so `mrt_named_color_alias` stands
        // for them here. Break the convention and they stop agreeing; that is the whole hazard.
        auto readers_agree = [&](const prosper::gpu::DrawItem& draw, uint32_t slot) {
            return same(mrt_pass_color_binding(draw, slot), mrt_color_binding(draw, slot)) &&
                   same(mrt_pass_color_binding(draw, slot), mrt_named_color_alias(draw, slot));
        };

        // Shape 1: the exact mirror. What `realize_draw_item` and every capture conversion emit.
        auto mirrored = make_draw();
        mirrored.color0_base = 0x300000ull;
        mirrored.color0_width = 32u; mirrored.color0_height = 2u;
        mirrored.color_targets[0] = {0x300000ull, 32u, 2u};
        mirrored.color1_base = 0x310000ull;
        mirrored.color1_width = 32u; mirrored.color1_height = 2u;
        mirrored.color_targets[1] = {0x310000ull, 32u, 2u};
        CHECK(mrt_color_alias_mirrored(mirrored, 0) && mrt_color_alias_mirrored(mirrored, 1));
        CHECK(readers_agree(mirrored, 0) && readers_agree(mirrored, 1));

        // Shape 2: the array ABSENT and the named triple authoritative -- captures through v33, and
        // the direct/synthetic callers the render fixtures are. The fallback repairs it, so the
        // readers still agree and this must NOT be reported as a divergence.
        auto legacy = make_draw();
        legacy.color0_base = 0x300000ull;
        legacy.color0_width = 32u; legacy.color0_height = 2u;
        CHECK(mrt_color_alias_mirrored(legacy, 0) && mrt_color_alias_mirrored(legacy, 1));
        CHECK(readers_agree(legacy, 0) && readers_agree(legacy, 1));

        // Shape 3: a draw that names no slot-0 surface at all. Both representations empty.
        auto unbound = make_draw();
        CHECK(mrt_color_alias_mirrored(unbound, 0) && mrt_color_alias_mirrored(unbound, 1));
        CHECK(readers_agree(unbound, 0) && readers_agree(unbound, 1));

        // Slots 2 and up have no named alias, so nothing there can diverge -- the array is the only
        // representation and is authoritative by itself.
        auto wide = make_draw();
        wide.color_targets[2].base = kMrt2;
        CHECK(mrt_color_alias_mirrored(wide, 2));

        // DIVERGENCE A -- a stale array entry beside a live named triple. This is the shape the
        // render fixtures reach by taking a captured item and updating only the named fields, and
        // the shape #3023's grouping fix was written against.
        auto stale_mirror = mirrored;
        stale_mirror.color_targets[0].base = 0x200000ull;   // an earlier target, never cleared
        stale_mirror.color_targets[0].width = 0u;
        stale_mirror.color_targets[0].height = 0u;
        CHECK(!mrt_color_alias_mirrored(stale_mirror, 0));
        CHECK(!readers_agree(stale_mirror, 0));
        // Named-first grouping and the renderer agree; the active-binding rule does not, so a
        // feedback test on this draw answers about a surface the pass never renders to.
        CHECK(mrt_pass_color_binding(stale_mirror, 0).base == 0x300000ull);
        CHECK(mrt_color_binding(stale_mirror, 0).base == 0x200000ull);

        // DIVERGENCE B -- the sharpest one, and the reason the unifying accessor is a semantic
        // decision rather than a refactor: here GROUPING and TARGET SELECTION disagree. The array
        // names a surface, the named triple names none, so grouping compares 0x500000 while
        // live_renderer's `pass_bases[0]` -- the raw named field -- is 0.
        auto array_only = make_draw();
        array_only.color_targets[0].base = 0x500000ull;
        CHECK(!mrt_color_alias_mirrored(array_only, 0));
        CHECK(!readers_agree(array_only, 0));
        CHECK(mrt_pass_color_binding(array_only, 0).base == 0x500000ull);
        CHECK(mrt_named_color_alias(array_only, 0).base == 0u);

        // DIVERGENCE C -- same address, different EXTENT. A base-only comparison would miss it, and
        // the extent is half of an attachment's identity: it is what stops two packed mip levels
        // sharing one Vulkan attachment.
        auto extent_only = mirrored;
        extent_only.color_targets[0].width = 16u;
        extent_only.color_targets[0].height = 16u;
        CHECK(!mrt_color_alias_mirrored(extent_only, 0));
        CHECK(mrt_color_alias_mirrored(extent_only, 1));   // slot 1 is untouched and still mirrors

        // DIVERGENCE D -- slot 1 diverges on its own. The resolve path takes its destination from
        // the raw named `color1_base` (#3025), so this slot carries the same hazard as slot 0.
        auto slot1_diverges = mirrored;
        slot1_diverges.color_targets[1].base = 0x999000ull;
        CHECK(!mrt_color_alias_mirrored(slot1_diverges, 1));
        CHECK(mrt_color_alias_mirrored(slot1_diverges, 0));

        // The consumer's entry point: it names the diverging slot and both representations, and it
        // counts. `test_gpu_capture_render` asserts the live renderer actually calls this, which is
        // the part a predicate test cannot establish.
        uint32_t reported_slot = 99u;
        Binding reported_carried{}, reported_named{};
        uint64_t reported_ordinal = 0u;
        const uint64_t before = mrt_color_alias_divergences();
        const uint32_t diverged = mrt_check_color_alias_mirror(
            stale_mirror, [&](uint32_t slot, const Binding& carried, const Binding& named,
                              uint64_t ordinal) {
                reported_slot = slot; reported_carried = carried; reported_named = named;
                reported_ordinal = ordinal;
            });
        CHECK(diverged == 1u && reported_slot == 0u && reported_ordinal == before + 1u);
        CHECK(reported_carried.base == 0x200000ull && reported_named.base == 0x300000ull &&
              reported_named.width == 32u && reported_named.height == 2u);
        CHECK(mrt_color_alias_divergences() == before + 1u);

        // ... and the three producer shapes report nothing, so the guard is not a constant.
        uint32_t spurious = 0;
        auto count_spurious = [&](uint32_t, const Binding&, const Binding&, uint64_t) {
            ++spurious;
        };
        CHECK(mrt_check_color_alias_mirror(mirrored, count_spurious) == 0u);
        CHECK(mrt_check_color_alias_mirror(legacy, count_spurious) == 0u);
        CHECK(mrt_check_color_alias_mirror(unbound, count_spurious) == 0u);
        CHECK(spurious == 0u && mrt_color_alias_divergences() == before + 1u);

        // Both slots of one draw are reported, once each.
        auto both_diverge = stale_mirror;
        both_diverge.color_targets[1].base = 0x999000ull;
        uint32_t both_reports = 0;
        CHECK(mrt_check_color_alias_mirror(
                  both_diverge, [&](uint32_t, const Binding&, const Binding&, uint64_t) {
                      ++both_reports; }) == 2u);
        CHECK(both_reports == 2u && mrt_color_alias_divergences() == before + 3u);
    }

    // #3025 -- the resolve DESTINATION is part of a resolve pass's identity.
    //
    // These arms establish the mechanism, not the outcome. The load-bearing assertion for this fix
    // is in test_gpu_capture_render ("a second resolve into a different destination receives the
    // resolved pixels"), which reads the destination surface's CONTENTS; a predicate that returns
    // false proves only that two draws are not grouped, not that both destinations were written.
    // What these arms add is the part that experiment cannot isolate: that the discriminating term
    // is the destination and nothing else.
    {
        // The exact shape live_renderer sees. A fixed-function resolve exports nothing, so its
        // color1_write_mask is 0 -- which is precisely why slot 1 could not discriminate.
        auto resolve = make_draw();
        resolve.ps.cb_resolve = true;
        resolve.color0_base = 0x700000ull;             // the MSAA scene being resolved
        resolve.color0_width = 32u; resolve.color0_height = 32u;
        resolve.color1_base = 0x710000ull;             // the single-sample destination
        resolve.color1_width = 32u; resolve.color1_height = 32u;
        resolve.ps.color_write_mask = 0xfu;
        resolve.ps.color1_write_mask = 0u;

        auto second = resolve;
        second.color1_base = 0x720000ull;              // a DIFFERENT destination

        // The premise, asserted rather than assumed: slot 1 is inactive on a resolve, so the
        // colour-identity predicate cannot tell these two apart. If this ever fails the fix below
        // is keying on a term that was already discriminating, and these arms would pass for the
        // wrong reason.
        CHECK(mrt_active_color(resolve, 1, format_defined) == 0u);
        CHECK(mrt_active_color_count(resolve, format_defined) == 1u);
        CHECK(mrt_same_color_pass(resolve, second, format_defined, format_at));

        CHECK(!mrt_same_resolve_pass(resolve, second));
        CHECK(mrt_same_resolve_pass(resolve, resolve));

        // A resolve never joins an ordinary draw's group: the copy ends the group, so an ordinary
        // draw grouped after one would be dropped.
        auto scene = resolve;
        scene.ps.cb_resolve = false;
        CHECK(!mrt_same_resolve_pass(resolve, scene));
        CHECK(!mrt_same_resolve_pass(scene, resolve));

        // Two ordinary draws are unaffected by the destination term, including when their stale
        // named color1_base differs. This is what keeps the change scoped to resolves: keying every
        // pass on a raw named field would split ordinary groups on stale register state.
        auto scene_other_c1 = scene;
        scene_other_c1.color1_base = 0x730000ull;
        CHECK(mrt_same_resolve_pass(scene, scene_other_c1));
    }

    if (failures == 0) std::printf("mrt_binding: OK\n");
    return failures == 0 ? 0 : 1;
}
