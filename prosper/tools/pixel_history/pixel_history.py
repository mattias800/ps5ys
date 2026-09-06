#!/usr/bin/env python3
"""Answer "why is this pixel this colour" from an RDC, in one call.

Usage: pixel_history.py CAPTURE --output NEW_DIRECTORY [--pixel X,Y] [--target N]
                        [--expect-control]

Four answers steer completely different investigations, and this separates them:

  NOTHING_DREW        no event in the frame touched the pixel -- look upstream at
                      geometry, viewport, culling or whether the draw was submitted
  ALL_REJECTED        fragments arrived and every one was thrown out; the report names
                      which test did it (depth, stencil, scissor, discard, cull, ...)
  SHADER_WROTE_BLACK  the shader ran, passed every test, and computed black -- look at
                      resource binding, textures, uniforms, the shader itself
  STORE_LOST_IT       the shader computed a non-black colour and the target ended black
                      anyway -- look at blend state, write masks, later overdraw

Run against `pixel_history_control` first on any new driver: `--expect-control` checks
this tool's own reading against a construction with a known answer.

Exit 0 a verdict was produced, 1 replay/analysis failed, 2 usage.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

# Rejections that happen BEFORE the fragment shader runs. RenderDoc still populates
# shaderOut for these, and the value is the previous event's residue -- measured on the
# known-answer control, where a scissored white draw reported the earlier draw's blue.
# Reporting that as "the shader computed blue" would be a confident lie, so these
# events carry no shader output at all.
# Rejections RenderDoc's Vulkan pixel history evaluates and returns on BEFORE running the
# fragment shader (vk_pixelhistory.cpp: scissor at :4470, sample mask at :4478, each
# `return`ing immediately). shaderOut is still populated for these and carries the previous
# event's residue -- see instrument trap 268.
PRE_FRAGMENT = ("scissorClipped", "backfaceCulled", "depthClipped", "viewClipped",
                "predicationSkipped", "sampleMasked")
# Rejections that normally happen AFTER the shader, so their shaderOut is real and is often
# the whole answer. The exception is a shader declaring EarlyFragmentTests, where RenderDoc
# evaluates these first (vk_pixelhistory.cpp:4525) and the output is residue again. The
# Python API exposes no per-event early-fragment flag, so these are reported with a caveat
# rather than suppressed -- and no verdict is ever derived from a rejected event's output.
POST_FRAGMENT = ("depthTestFailed", "stencilTestFailed", "depthBoundsFailed")
# Not a rejection at all: Passed() excludes unboundPS (data_types.h:2759), so such an event
# arrives passed=True while its own documentation says the output "may have undefined
# values" (:2664). Trusting it is trap 268 one list entry over, so it is called out by name
# and its output is never used.
UNDEFINED_OUTPUT = ("unboundPS",)
REJECTIONS = PRE_FRAGMENT + POST_FRAGMENT + ("shaderDiscarded",) + UNDEFINED_OUTPUT
# Vulkan-only note: predicationSkipped is set only by the D3D11 backend, and on D3D11/12 the
# sample mask is the output-merger mask applied AFTER the shader. This tool reads prosper's
# Vulkan captures; a D3D11 capture would need sampleMasked moved out of PRE_FRAGMENT.
BLACK = 1.0 / 255.0


def suppress_shader_output(rejected_by):
    """Is this event's shaderOut another draw's residue, or undefined?

    Extracted from the replay path deliberately: that path needs RenderDoc and a GPU, so
    nothing in CI can reach it. Leaving the decision inline meant deleting PRE_FRAGMENT
    left every test green while the tool began reporting a neighbouring draw's colour as
    this draw's shader output -- measured, not hypothetical.
    """
    return any(f in PRE_FRAGMENT or f in UNDEFINED_OUTPUT for f in rejected_by)


def classify(events):
    """One verdict from the event list. Order matters: the earliest true statement wins."""
    if not events:
        return "NOTHING_DREW", "No event in this frame touched the pixel."
    passed = [e for e in events if e["passed"]]
    if not passed:
        why = {}
        for e in events:
            for r in e["rejected_by"]:
                why[r] = why.get(r, 0) + 1
        named = ", ".join(f"{k}x{v}" for k, v in sorted(why.items(), key=lambda kv: -kv[1]))
        return "ALL_REJECTED", f"{len(events)} events reached the pixel, none survived: {named or 'no reason flagged'}."

    # Only the LAST passing event can explain the pixel's final state. An earlier bright
    # writer that was legitimately overdrawn -- a white sky behind a black object -- is not
    # evidence of a lost store, and reading it as one made SHADER_WROTE_BLACK unreachable
    # on any pixel with history, which is most of a real frame.
    final = passed[-1]
    lit = max(final["postMod"][:3]) if final["postMod"] else 0.0
    if lit > BLACK:
        return "PIXEL_WAS_WRITTEN", f"{len(passed)} of {len(events)} events passed; final colour is not black."
    if final["shader_output_suppressed"]:
        named = ", ".join(final["rejected_by"]) or "unknown"
        return "OUTPUT_UNTRUSTED", (
            f"the last passing event ({named}) has no trustworthy shader output, so "
            f"'computed black' and 'store lost it' cannot be separated here -- pick "
            f"another pixel, or inspect that event directly.")
    if final["shaderOut"] and max(final["shaderOut"][:3]) > BLACK:
        return "STORE_LOST_IT", (
            "the last passing event computed a non-black colour and the target is black "
            "anyway -- blend state, write mask, or a later event that is not in this list.")
    return "SHADER_WROTE_BLACK", (
        f"{len(passed)} event(s) passed every test and the last one computed black -- "
        f"resource binding, textures, uniforms or the shader itself.")


def embedded():
    req = json.loads(Path(os.environ["PROSPER_PIXHIST_REQUEST"]).read_text())
    out = Path(req["output"])
    result = {"status": "FAILED"}
    try:
        import renderdoc as rd

        def values(mv):
            col = getattr(mv, "col", None)
            v = list(getattr(col, "floatValue", []) or []) if col is not None else []
            return [float(x) for x in v][:4]

        cap = rd.OpenCaptureFile()
        if cap.OpenFile(req["capture"], "", None) != rd.ResultCode.Succeeded:
            raise RuntimeError("OpenFile failed; not an RDC this build can read")
        st, ctl = cap.OpenCapture(rd.ReplayOptions(), None)
        if st != rd.ResultCode.Succeeded:
            raise RuntimeError(f"OpenCapture failed: {st}")

        # Ask the API whether it supports this at all. Without it an unsupported driver
        # returns an empty history, which is byte-identical to "nothing drew here" -- the
        # exact confusion this tool exists to remove.
        props = ctl.GetAPIProperties()
        if not props.pixelHistory:
            raise RuntimeError(
                "this replay driver reports no pixel-history support; an empty result here "
                "would be indistinguishable from 'nothing drew', so refusing to report one")

        actions = []

        def walk(nodes):
            for a in nodes:
                if a.flags & rd.ActionFlags.Drawcall:
                    actions.append(a)
                walk(a.children)

        walk(ctl.GetRootActions())
        if not actions:
            raise RuntimeError("no draw actions in capture")
        ctl.SetFrameEvent(actions[-1].eventId, True)

        # Prefer the targets actually BOUND at the selected event. TextureCategory.ColorTarget
        # is set for VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT as well as COLOR_ATTACHMENT
        # (vk_info.cpp:2601-2603), so scanning every texture can hand back a transient
        # attachment nothing presents -- and its empty history reads as NOTHING_DREW.
        by_id = {t.resourceId: t for t in ctl.GetTextures()}
        bound = [by_id[o.resource] for o in ctl.GetPipelineState().GetOutputTargets()
                 if o.resource != rd.ResourceId.Null() and o.resource in by_id]
        source, targets = "bound output targets at the last draw", bound
        if not targets:
            targets = [t for t in ctl.GetTextures()
                       if t.creationFlags & rd.TextureCategory.ColorTarget]
            source = "texture scan (no bound output target; may include transient attachments)"
        if not targets:
            raise RuntimeError("no colour targets in capture")
        if not 0 <= req["target"] < len(targets):
            raise RuntimeError(
                f"--target {req['target']} out of range: {len(targets)} target(s) available "
                f"from {source}. Silently clamping would analyse a different image than asked.")
        tex = targets[req["target"]]

        # Save the target first: selection needs to see the image, and the image is also
        # the evidence a reader wants beside the verdict.
        image = str(out / "target.png")
        save = rd.TextureSave()
        save.resourceId = tex.resourceId
        save.destType = rd.FileType.PNG
        if ctl.SaveTexture(save, image) != rd.ResultCode.Succeeded:
            image = None

        pixel, how = req["pixel"], "requested"
        if pixel is None:
            pixel, how = (tex.width // 2, tex.height // 2), "centre (no content guidance)"
            if image:
                try:
                    from PIL import Image
                    im = Image.open(image).convert("RGB")
                    # The brightest pixel, not the centre: on a mostly-black frame the
                    # centre is exactly where nothing happened, and its empty history
                    # says nothing about why the frame is black.
                    small = im.resize((min(im.width, 256), min(im.height, 256)))
                    sx, sy = im.width / small.width, im.height / small.height
                    best, bxy = -1, None
                    for y in range(small.height):
                        for x in range(small.width):
                            r_, g_, b_ = small.getpixel((x, y))
                            lum = r_ * 2 + g_ * 5 + b_
                            if lum > best:
                                best, bxy = lum, (x, y)
                    pixel = (int(bxy[0] * sx), int(bxy[1] * sy))
                    how = f"brightest pixel (luma {best})"
                except Exception as exc:
                    how = f"centre (content guidance unavailable: {exc})"

        if not (0 <= pixel[0] < tex.width and 0 <= pixel[1] < tex.height):
            raise RuntimeError(
                f"pixel {tuple(pixel)} is outside the {tex.width}x{tex.height} target; an "
                f"out-of-bounds history is empty and would read as NOTHING_DREW")
        hist = ctl.PixelHistory(tex.resourceId, pixel[0], pixel[1],
                                rd.Subresource(0, 0, 0), rd.CompType.Typeless)
        events = []
        for m in hist:
            rejected = [f for f in REJECTIONS if getattr(m, f, False)]
            pre = suppress_shader_output(rejected)
            events.append({
                "eventId": int(m.eventId),
                "passed": bool(m.Passed()),
                "rejected_by": rejected,
                # Suppressed rather than reported as zero: "the shader ran and produced
                # nothing" and "the shader never ran" are different findings.
                "shaderOut": None if pre else values(m.shaderOut),
                "shader_output_suppressed": pre,
                "postMod": values(m.postMod),
            })
        verdict, reason = classify(events)

        control = {}
        if req.get("expect_control"):
            for name, (cx, cy), _ in CONTROL_REGIONS:
                ch = ctl.PixelHistory(tex.resourceId, cx, cy, rd.Subresource(0, 0, 0),
                                      rd.CompType.Typeless)
                ce = []
                for m in ch:
                    rej = [f for f in REJECTIONS if getattr(m, f, False)]
                    pre = suppress_shader_output(rej)
                    ce.append({"eventId": int(m.eventId), "passed": bool(m.Passed()),
                               "rejected_by": rej,
                               "shaderOut": None if pre else values(m.shaderOut),
                               "shader_output_suppressed": pre,
                               "postMod": values(m.postMod)})
                control[name] = {"verdict": classify(ce)[0], "events": ce}

        result = {"status": "REPLAYED", "verdict": verdict, "reason": reason,
                  "control_regions": control,
                  "pixel": list(pixel), "pixel_choice": how,
                  "target": {"id": str(tex.resourceId), "width": tex.width,
                             "height": tex.height, "chosen_from": source,
                             "candidates": len(targets)},
                  "target_image": image, "draw_count": len(actions),
                  "events": events,
                  "api": {"pixelHistory": bool(props.pixelHistory),
                          "shaderDebugging": bool(props.shaderDebugging)}}
    except Exception:
        result = {"status": "FAILED", "error": traceback.format_exc()}
    (out / "pixel_history.json").write_text(json.dumps(result, indent=2) + "\n")
    # qrenderdoc would otherwise enter its UI event loop after the startup script.
    os._exit(0 if result["status"] == "REPLAYED" else 1)


# The control's five regions, restated here independently of the C source that builds them.
# Each exists to construct ONE verdict: a control that only ever produces one tests the
# machinery and not the distinctions the tool is for.
CONTROL_REGIONS = [
    ("A  sequence", (16, 16), "PIXEL_WAS_WRITTEN"),
    ("A' arm-1 only", (16, 36), "PIXEL_WAS_WRITTEN"),
    ("B  black draw", (48, 16), "SHADER_WROTE_BLACK"),
    ("C  no write", (16, 48), "STORE_LOST_IT"),
    ("E  all killed", (48, 48), "ALL_REJECTED"),
]


def check_control(regions):
    """Check a full control reading: every verdict, and the suppression trap 268 needs.

    `regions` maps region name -> {"verdict": str, "events": [...]}.
    """
    problems = []
    for name, _, want in CONTROL_REGIONS:
        got = regions.get(name)
        if got is None:
            problems.append(f"{name}: region missing from the reading")
        elif got["verdict"] != want:
            problems.append(f"{name}: expected {want}, read {got['verdict']}")

    seq = regions.get("A  sequence")
    if seq:
        reasons = [r for e in seq["events"] for r in e["rejected_by"]]
        for needed in ("depthTestFailed", "shaderDiscarded", "scissorClipped"):
            if needed not in reasons:
                problems.append(f"A: {needed} was constructed but not reported; a rejection "
                                f"is being misattributed")
        # Trap 268: the scissored arm's shaderOut is the PREVIOUS draw's colour. If any
        # positionally-rejected event still carries an output, the suppression is off and the
        # tool is one step from reporting a neighbouring draw's colour as this one's.
        leaked = [e["eventId"] for e in seq["events"]
                  if suppress_shader_output(e["rejected_by"]) and e["shaderOut"] is not None]
        if leaked:
            problems.append(f"A: events {leaked} were rejected before the fragment shader ran "
                            f"and still reported a shader output; the stale-value suppression "
                            f"is not working")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixel", help="X,Y; default is the brightest pixel in the target")
    parser.add_argument("--target", type=int, default=0, help="colour target index")
    parser.add_argument("--expect-control", action="store_true",
                        help="check this tool's reading against pixel_history_control")
    args = parser.parse_args()
    if not args.capture.is_file():
        parser.error("capture does not exist")
    if not shutil.which("qrenderdoc"):
        parser.error("qrenderdoc is not installed in this environment")
    pixel = None
    if args.pixel:
        try:
            x, y = (int(v) for v in args.pixel.split(","))
            pixel = [x, y]
        except ValueError:
            parser.error("--pixel must be X,Y")
    try:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(f"output must be a new directory: {exc}")

    request = {"capture": str(args.capture.resolve()), "output": str(output),
               "pixel": pixel, "target": args.target,
               "expect_control": args.expect_control}
    request_path = output / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    env = {**os.environ, "PROSPER_PIXHIST_REQUEST": str(request_path),
           "QT_QPA_PLATFORM": "offscreen"}
    try:
        proc = subprocess.run(["qrenderdoc", "--python", str(Path(__file__).resolve())],
                              env=env, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        (output / "process.log").write_text(str(exc))
        print(f"Replay unavailable: {exc}", file=sys.stderr)
        return 1
    (output / "process.log").write_text(proc.stdout + proc.stderr)
    report = output / "pixel_history.json"
    if not report.is_file():
        print(f"Replay produced no report (exit={proc.returncode}); inspect {output}",
              file=sys.stderr)
        return 1
    data = json.loads(report.read_text())
    if data.get("status") != "REPLAYED":
        print(f"Analysis failed; inspect {output}\n{data.get('error', '')}", file=sys.stderr)
        return 1

    if args.expect_control:
        regions = data.get("control_regions") or {}
        problems = check_control(regions)
        for name, _, _ in CONTROL_REGIONS:
            got = regions.get(name)
            print(f"  {name:<16} {got['verdict'] if got else 'MISSING'}")
        for problem in problems:
            print(f"CONTROL FAILED: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"CONTROL VERIFIED: {len(CONTROL_REGIONS)} regions, every constructed verdict "
              f"and rejection reason read back correctly.")

    print(f"{data['verdict']}: {data['reason']}")
    print(f"  pixel {tuple(data['pixel'])} of {data['target']['width']}x"
          f"{data['target']['height']}, chosen by {data['pixel_choice']}")
    print(f"  target {data['target']['id']} of {data['target']['candidates']}, "
          f"from {data['target']['chosen_from']}")
    print(f"  {data['draw_count']} draws in frame, {len(data['events'])} touched this pixel")
    for e in data["events"]:
        out = ("suppressed (no fragment ran)" if e["shader_output_suppressed"]
               else ("[" + ", ".join(f"{v:.3f}" for v in e["shaderOut"]) + "]"
                     if e["shaderOut"] else "-"))
        # Passed() ignores unboundPS, so an event with no bound pixel shader arrives as a
        # pass. Printing a bare "PASS" would hide the one fact that matters about it.
        undefined = [f for f in e["rejected_by"] if f in UNDEFINED_OUTPUT]
        state = (("PASS!" + ",".join(undefined)) if e["passed"] and undefined
                 else "PASS" if e["passed"]
                 else ",".join(e["rejected_by"]) or "rejected")
        print(f"    eid {e['eventId']:>6}  {state:<20} shaderOut {out}")
    print(f"  evidence: {args.output}")
    return 0


if os.environ.get("PROSPER_PIXHIST_REQUEST") and "pyrenderdoc" in globals():
    embedded()
elif __name__ == "__main__":
    sys.exit(main())
