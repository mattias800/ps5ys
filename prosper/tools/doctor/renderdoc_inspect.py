#!/usr/bin/env python3
"""Replay an RDC and export actions, buffer/texture inventory and the final color target.

Usage: renderdoc_inspect.py CAPTURE --output NEW_DIRECTORY [--expect-index-control]
Uses qrenderdoc's embedded Python bindings when the distribution provides no importable module.
Exit 0 requires successful GPU replay and at least one draw; index control additionally checks
the final SSBO's exact bytes. This does not assert correctness of arbitrary game captures.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import traceback


def index_control_matches(actions, buffers, read_buffer):
    expected = struct.pack("<256I", *range(6), *([0xffffffff] * 250))
    matched = [str(buffer.resourceId) for buffer in buffers
               if buffer.length == len(expected)
               and bytes(read_buffer(buffer.resourceId, 0, buffer.length)) == expected]
    if len(actions) != 5 or len(matched) != 1:
        raise RuntimeError(f"Index control mismatch: draws={len(actions)}, "
                           f"exact SSBO matches={len(matched)} (expected 5 and 1).")
    return matched


def inspect(request):
    import renderdoc as rd
    capture = rd.OpenCaptureFile()
    controller = None
    try:
        status = capture.OpenFile(request["capture"], "", None)
        if status != rd.ResultCode.Succeeded:
            raise RuntimeError(f"OpenFile: {status}")
        status, controller = capture.OpenCapture(rd.ReplayOptions(), None)
        if status != rd.ResultCode.Succeeded:
            raise RuntimeError(f"OpenCapture: {status}")
        actions = []

        def walk(nodes):
            for action in nodes:
                if action.flags & rd.ActionFlags.Drawcall:
                    actions.append(action)
                walk(action.children)

        walk(controller.GetRootActions())
        if not actions:
            raise RuntimeError("No draw actions in capture; not a graphics replay control.")
        controller.SetFrameEvent(actions[-1].eventId, True)
        buffers = list(controller.GetBuffers())
        textures = list(controller.GetTextures())
        matched = []
        if request["expect_index_control"]:
            matched = index_control_matches(actions, buffers, controller.GetBufferData)
        outputs = controller.GetPipelineState().GetOutputTargets()
        image_path = None
        if outputs and outputs[0].resource != rd.ResourceId.Null():
            save = rd.TextureSave()
            save.resourceId = outputs[0].resource
            save.destType = rd.FileType.PNG
            image_path = str(Path(request["output"]) / "final-target.png")
            status = controller.SaveTexture(save, image_path)
            if status != rd.ResultCode.Succeeded:
                raise RuntimeError(f"SaveTexture: {status}")
        return {
            "status": "REPLAYED", "api": str(controller.GetAPIProperties().pipelineType),
            "draws": [{"event": a.eventId, "indices": a.numIndices, "instances": a.numInstances}
                      for a in actions],
            "buffers": [{"id": str(b.resourceId), "bytes": b.length} for b in buffers],
            "textures": [{"id": str(t.resourceId), "width": t.width, "height": t.height}
                         for t in textures],
            "final_target": image_path, "index_control_ssbo": matched,
        }
    finally:
        if controller is not None:
            controller.Shutdown()
        capture.Shutdown()


def embedded():
    request = json.loads(Path(os.environ["PROSPER_RDOC_REQUEST"]).read_text())
    code = 0
    try:
        result = inspect(request)
    except Exception:
        code = 1
        result = {"status": "FAILED", "error": traceback.format_exc()}
    Path(request["output"], "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    # qrenderdoc would otherwise enter its UI event loop after the startup script.
    # Replay handles above are shut down explicitly and the report is closed before exit.
    os._exit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-index-control", action="store_true")
    args = parser.parse_args()
    if not args.capture.is_file():
        parser.error("capture does not exist")
    if not shutil.which("qrenderdoc"):
        parser.error("qrenderdoc is not installed in this environment")
    try:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(f"output must be a new directory: {exc}")
    request = {"capture": str(args.capture.resolve()), "output": str(output),
               "expect_index_control": args.expect_index_control}
    request_path = output / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    env = {**os.environ, "PROSPER_RDOC_REQUEST": str(request_path), "QT_QPA_PLATFORM": "offscreen"}
    try:
        result = subprocess.run(["qrenderdoc", "--python", str(Path(__file__).resolve())],
                                env=env, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        (output / "process.log").write_text(str(exc))
        print(f"Replay unavailable: {exc}", file=sys.stderr)
        return 1
    (output / "process.log").write_text(result.stdout + result.stderr)
    report_path = output / "replay.json"
    if result.returncode != 0 or not report_path.is_file():
        print(f"Replay failed (exit={result.returncode}); inspect {output}", file=sys.stderr)
        return 1
    try:
        report = json.loads(report_path.read_text())
        if not isinstance(report, dict) or report.get("status") != "REPLAYED":
            raise ValueError("report does not certify replay")
    except (OSError, ValueError) as exc:
        print(f"Invalid replay report: {exc}; inspect {output}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if os.environ.get("PROSPER_RDOC_REQUEST") and "pyrenderdoc" in globals():
    embedded()
elif __name__ == "__main__":
    sys.exit(main())
