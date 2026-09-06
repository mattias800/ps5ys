#!/usr/bin/env python3
"""The verdict must not soften: a black frame has four causes and they are not interchangeable."""
import unittest
import pixel_history as ph


def clear(post=(0.0, 0.0, 0.0, 1.0), eid=0, pre=(0.5, 0.5, 0.5, 1.0)):
    """A clear: passes, evaluates no test, and is never the subject of a verdict.

    `pre` differs from `post` by default, i.e. a clear that actually changed the pixel.
    """
    return ev(True, post=post, eid=eid, is_clear=True, pre=pre)


def ev(passed, rejected=(), shader=None, post=(0.0, 0.0, 0.0, 1.0), eid=1, suppressed=None,
       is_clear=False, pre=None):
    if suppressed is None:
        suppressed = ph.suppress_shader_output(rejected)
    return {"eventId": eid, "passed": passed, "rejected_by": list(rejected),
            "is_clear": is_clear, "preMod": list(pre if pre is not None else post),
            "shaderOut": None if suppressed else (None if shader is None else list(shader)),
            "shader_output_suppressed": suppressed,
            "postMod": list(post)}


class ClassifyTests(unittest.TestCase):
    def test_a_cleared_target_that_nothing_drew_to_is_not_a_shader_defect(self):
        # THE headline case: a black region of a cleared target. The clear passes and its
        # postMod is black, so reading it as the explaining event returned SHADER_WROTE_BLACK
        # and sent the reader to a shader that never ran.
        v, why = ph.classify([clear()])
        self.assertEqual(v, "NOTHING_DREW")
        self.assertIn("clear", why)

    def test_a_coloured_clear_that_nothing_drew_to_is_also_nothing_drew(self):
        # The same defect in its other direction: a sky-blue clear read as PIXEL_WAS_WRITTEN,
        # retiring the question with "something drew here" when nothing did.
        self.assertEqual(ph.classify([clear(post=(0.3, 0.6, 0.9, 1))])[0], "NOTHING_DREW")

    def test_a_clear_does_not_rescue_a_pixel_whose_draws_all_failed(self):
        v, why = ph.classify([clear(), ev(False, ["depthTestFailed"], eid=5)])
        self.assertEqual(v, "ALL_REJECTED")
        self.assertIn("1 draw", why)

    def test_a_clear_after_the_last_draw_is_the_thing_you_are_looking_at(self):
        v, why = ph.classify([ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1), eid=5),
                              clear(eid=9)])
        self.assertEqual(v, "CLEARED_AFTER_DRAW")
        self.assertIn("9", why)

    def test_a_clear_that_changed_nothing_does_not_take_the_blame(self):
        # A late clear whose preMod equals its postMod explains nothing. Blaming it would
        # move the reader off the draw that actually produced the pixel.
        v, _ = ph.classify([ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=5),
                            clear(post=(0, 0, 0, 1), pre=(0, 0, 0, 1), eid=9)])
        self.assertEqual(v, "SHADER_WROTE_BLACK")

    def test_a_clear_before_the_draws_does_not_change_a_real_verdict(self):
        drawn = ev(True, shader=(0.7, 0.2, 0.1, 1), post=(0, 0, 0, 1), eid=5)
        self.assertEqual(ph.classify([clear(), drawn])[0], "STORE_LOST_IT")

    def test_no_events_is_not_a_rendering_bug(self):
        v, why = ph.classify([])
        self.assertEqual(v, "NOTHING_DREW")
        self.assertIn("No event", why)

    def test_all_rejected_names_the_test_that_did_it(self):
        v, why = ph.classify([ev(False, ["depthTestFailed"]), ev(False, ["depthTestFailed"]),
                              ev(False, ["scissorClipped"])])
        self.assertEqual(v, "ALL_REJECTED")
        self.assertIn("depthTestFailedx2", why)
        self.assertIn("scissorClippedx1", why)

    def test_written_pixel_is_not_a_defect(self):
        v, _ = ph.classify([ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1))])
        self.assertEqual(v, "PIXEL_WAS_WRITTEN")

    def test_shader_black_and_store_lost_are_distinguished(self):
        # Same final pixel, same pass/fail pattern, opposite investigations.
        black = ph.classify([ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1))])[0]
        lost = ph.classify([ev(True, shader=(0.7, 0.2, 0.1, 1), post=(0, 0, 0, 1))])[0]
        self.assertEqual(black, "SHADER_WROTE_BLACK")
        self.assertEqual(lost, "STORE_LOST_IT")

    def test_stale_shader_output_cannot_manufacture_store_lost_it(self):
        # Trap 268 end to end: the scissored event carries the PREVIOUS draw's bright
        # colour. `suppressed=False` forces the pre-fix behaviour into the fixture, so this
        # fails unless classify() refuses to derive a verdict from a rejected event.
        stale = ev(False, ["scissorClipped"], shader=(0.9, 0.9, 0.9, 1), suppressed=False)
        passing_black = ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1))
        v, _ = ph.classify([stale, passing_black])
        self.assertEqual(v, "SHADER_WROTE_BLACK")

    def test_legitimately_overdrawn_bright_writer_is_not_a_lost_store(self):
        # A white sky, then a black object correctly drawn in front of it. Reading ANY
        # passing event made SHADER_WROTE_BLACK unreachable on every pixel with history --
        # which is most of a real frame, and precisely the frames people investigate.
        sky = ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1), eid=10)
        obj = ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1), eid=20)
        self.assertEqual(ph.classify([sky, obj])[0], "SHADER_WROTE_BLACK")

    def test_store_lost_it_still_reachable_after_an_overdraw(self):
        sky = ev(True, shader=(1, 1, 1, 1), post=(1, 1, 1, 1), eid=10)
        lost = ev(True, shader=(0.6, 0.3, 0.1, 1), post=(0, 0, 0, 1), eid=20)
        self.assertEqual(ph.classify([sky, lost])[0], "STORE_LOST_IT")

    def test_unbound_pixel_shader_is_not_a_clean_pass(self):
        # Passed() excludes unboundPS (data_types.h:2759), so the event arrives passed=True
        # while its output is documented as undefined. Deriving STORE_LOST_IT from that is
        # trap 268 one list entry over.
        self.assertTrue(ph.suppress_shader_output(["unboundPS"]))
        v, why = ph.classify([ev(True, ["unboundPS"], shader=(0.8, 0.8, 0.8, 1),
                                  post=(0, 0, 0, 1))])
        self.assertEqual(v, "OUTPUT_UNTRUSTED")
        self.assertIn("unboundPS", why)

    def test_a_rejected_event_never_counts_as_the_final_colour(self):
        v, _ = ph.classify([ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1)),
                            ev(False, ["depthTestFailed"], shader=(1, 1, 1, 1),
                               post=(1, 1, 1, 1))])
        self.assertEqual(v, "SHADER_WROTE_BLACK")

    def test_near_black_is_black(self):
        # A pixel at 1/255 is black to a viewer; a threshold of exactly 0 would call
        # every dithered or rounded frame "written" and retire the question wrongly.
        v, _ = ph.classify([ev(True, shader=(0.002, 0.001, 0.0, 1), post=(0.002, 0, 0, 1))])
        self.assertEqual(v, "SHADER_WROTE_BLACK")


class ControlCheckTests(unittest.TestCase):
    """check_control must fail on a wrong READING, not merely on a wrong final picture."""

    def regions(self):
        seq = [clear(eid=0),
               ev(True, shader=(0, 1, 0, 1), post=(0, 1, 0, 1), eid=1),
               ev(False, ["scissorClipped"], eid=2),
               ev(False, ["shaderDiscarded"], shader=(0, 0, 0, 0), eid=3),
               ev(True, shader=(1, 1, 0, 1), post=(1, 1, 0, 1), eid=4),
               ev(False, ["depthTestFailed"], shader=(0, 0, 1, 1), eid=5),
               ev(False, ["scissorClipped"], eid=9)]   # a later region's draw
        killed = [clear(eid=0), clear(eid=3), ev(False, ["scissorClipped"], eid=4),
                  ev(False, ["shaderDiscarded"], shader=(0, 0, 0, 0), eid=8)]
        return {"A  sequence": {"verdict": "PIXEL_WAS_WRITTEN", "events": seq},
                "A' arm-1 only": {"verdict": "PIXEL_WAS_WRITTEN", "events": []},
                "B  black draw": {"verdict": "SHADER_WROTE_BLACK", "events": []},
                "C  no write": {"verdict": "STORE_LOST_IT", "events": []},
                "E  all killed": {"verdict": "ALL_REJECTED", "events": killed}}

    def test_As_own_scissor_arm_must_precede_its_last_surviving_draw(self):
        # Deleting arm 2 leaves a later region's scissorClipped in A's history, so a check
        # that only asks "is scissorClipped present" stays green with the arm gone.
        bad = self.regions()
        bad["A  sequence"]["events"] = [e for e in bad["A  sequence"]["events"]
                                        if e["eventId"] != 2]
        self.assertIn("arm 2", " ".join(ph.check_control(bad)))

    def test_E_must_contain_its_own_discarded_draw(self):
        bad = self.regions()
        bad["E  all killed"]["events"] = [e for e in bad["E  all killed"]["events"]
                                          if "shaderDiscarded" not in e["rejected_by"]]
        self.assertIn("region's own draw is missing", " ".join(ph.check_control(bad)))

    def test_E_must_contain_both_clear_forms(self):
        # One clear is not enough: the transfer clear carries ActionFlags::Clear and the
        # loadOp clear does not, so a reading with only the flagged one stays green while
        # every loadOp clear in a real capture goes unrecognised.
        for keep in (0, 1):
            bad = self.regions()
            cs = [e for e in bad["E  all killed"]["events"] if e["is_clear"]]
            bad["E  all killed"]["events"] = [
                e for e in bad["E  all killed"]["events"]
                if not e["is_clear"] or e["eventId"] == cs[keep]["eventId"]]
            self.assertIn("expected 2", " ".join(ph.check_control(bad)))

    def test_correct_reading_passes(self):
        self.assertEqual(ph.check_control(self.regions()), [])

    def test_every_region_verdict_is_checked(self):
        # Each region exists to construct one verdict; a tool that collapsed them all to the
        # same answer would still satisfy a check that only looked at one.
        for name in ("A  sequence", "B  black draw", "C  no write", "E  all killed"):
            bad = self.regions()
            bad[name]["verdict"] = "NOTHING_DREW"
            self.assertTrue(ph.check_control(bad), name)

    def test_missing_region_is_caught(self):
        bad = self.regions()
        del bad["C  no write"]
        self.assertIn("missing", " ".join(ph.check_control(bad)))

    def test_misattributed_rejection_is_caught(self):
        # The reading stays well-formed and the verdict stays right; only the REASON is
        # wrong. That is the failure a final-picture check cannot see.
        bad = self.regions()
        # Target by event id, not position: the fixture's indices move whenever the
        # construction gains an event, and a positional fixture silently retargets.
        bad["A  sequence"]["events"] = [
            ev(False, ["stencilTestFailed"], shader=(0, 0, 1, 1), eid=5)
            if e["eventId"] == 5 else e
            for e in bad["A  sequence"]["events"]]
        self.assertIn("depthTestFailed", " ".join(ph.check_control(bad)))

    def test_unsuppressed_stale_output_is_caught(self):
        bad = self.regions()
        bad["A  sequence"]["events"][1] = ev(False, ["scissorClipped"], shader=(0, 0, 1, 1),
                                             eid=2, suppressed=False)
        self.assertIn("stale-value suppression", " ".join(ph.check_control(bad)))


class SuppressionTests(unittest.TestCase):
    """Guards the decision the replay path makes; CI has no GPU, so this is its only cover."""

    def test_pre_fragment_rejections_suppress(self):
        for reason in ("scissorClipped", "backfaceCulled", "depthClipped", "viewClipped",
                       "predicationSkipped", "sampleMasked"):
            self.assertTrue(ph.suppress_shader_output([reason]), reason)

    def test_post_fragment_rejections_keep_their_output(self):
        # The shader DID run for these, so its output is real and often the whole answer:
        # a depth-failed draw that computed the missing colour names the defect.
        for reason in ("depthTestFailed", "stencilTestFailed", "shaderDiscarded",
                       "depthBoundsFailed"):
            self.assertFalse(ph.suppress_shader_output([reason]), reason)

    def test_passing_event_keeps_its_output(self):
        self.assertFalse(ph.suppress_shader_output([]))

    def test_mixed_rejection_suppresses(self):
        self.assertTrue(ph.suppress_shader_output(["depthTestFailed", "scissorClipped"]))

if __name__ == "__main__":
    unittest.main()
