#!/usr/bin/env python3
"""The verdict must not soften: a black frame has four causes and they are not interchangeable."""
import unittest
import pixel_history as ph


def ev(passed, rejected=(), shader=None, post=(0.0, 0.0, 0.0, 1.0), eid=1):
    return {"eventId": eid, "passed": passed, "rejected_by": list(rejected),
            "shaderOut": None if shader is None else list(shader),
            "shader_output_suppressed": shader is None and bool(rejected),
            "postMod": list(post)}


class ClassifyTests(unittest.TestCase):
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
        # The trap the control caught: a scissored event carries the PREVIOUS draw's
        # colour. If that value were trusted, a frame where nothing survived would be
        # reported as a blend/write-mask defect and send the reader to the wrong file.
        passing_black = ev(True, shader=(0, 0, 0, 1), post=(0, 0, 0, 1))
        scissored = ev(False, ["scissorClipped"], shader=None, post=(0, 0, 0, 1))
        v, _ = ph.classify([scissored, passing_black])
        self.assertEqual(v, "SHADER_WROTE_BLACK")

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
    def good(self):
        return [ev(True, post=(1, 0, 0, 1)), ev(True, shader=(0, 1, 0, 1), post=(0, 1, 0, 1)),
                ev(False, ["depthTestFailed"], shader=(0, 0, 1, 1)),
                ev(False, ["scissorClipped"], shader=None),
                ev(False, ["shaderDiscarded"], shader=(0, 0, 0, 0)),
                ev(True, shader=(1, 1, 0, 1), post=(1, 1, 0, 1))]

    def test_correct_reading_passes(self):
        self.assertEqual(ph.check_control(self.good()), [])

    def test_wrong_outcome_order_is_caught(self):
        bad = self.good()
        bad[2], bad[3] = bad[3], bad[2]
        self.assertTrue(ph.check_control(bad))

    def test_unsuppressed_stale_output_is_caught(self):
        bad = self.good()
        bad[3] = ev(False, ["scissorClipped"], shader=(0, 0, 1, 1))
        self.assertIn("stale-value suppression", " ".join(ph.check_control(bad)))

    def test_wrong_final_colour_is_caught(self):
        bad = self.good()
        bad[-1] = ev(True, shader=(1, 1, 0, 1), post=(1, 1, 1, 1))
        self.assertTrue(ph.check_control(bad))



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
