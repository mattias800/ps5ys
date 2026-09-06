# Diagnostic workflow checks

This directory checks whether an external instrument can produce usable evidence in the
current environment. It owns bounded positive controls, capability reports, and the small
standalone build used to run existing host tests under sanitizers. It does not own guest
diagnostics or game benchmarks; reuse the tools linked in
`docs/DEBUGGING_WORKFLOWS.md` for those.

A discovered executable is only installed. A successful command without the expected
observation is inconclusive. Controls must retain command, exit status and output, distinguish
unavailable from ready, and use new disk-backed artifact directories. Never change privileges
or attach to an unrelated process automatically.

**One capability, one verdict.** Capabilities that can be denied independently are probed
independently, even when one tool provides them: user-space and kernel sampling are separate
probes because `perf_event_paranoid=2` denies the second and permits the first. Bundling them
made a usable host report UNAVAILABLE — a false negative is quieter than a false positive, and
so is likelier to survive, but it still answers a question nobody asked.

The scheduler capability check is explicitly system-wide: wakeup tracepoints can occur outside
the target's recording context. Require `--system-wide-scheduler` before collecting other
processes' events, limit the recording to two seconds, and never publish raw system-wide traces.
CPU and debugger controls remain restricted to their own children. Successful capability
controls do not certify loss-free game captures or attribution to a particular game thread.
