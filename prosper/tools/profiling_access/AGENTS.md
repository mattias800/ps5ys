# Autonomous profiling access

This is a host privilege boundary, not emulator code. The installed helper is a root-owned,
isolated-Python copy outside the repository. Only fixed scheduler events, a bounded duration,
and output through an already-open stdout are allowed. Never add arbitrary command execution,
caller-selected perf options, output paths, Python imports, or config files to the root helper.

The installer is administrator-only; it must validate sudoers before enabling access, refuse
unknown existing files, and revoke access before removing the helper. Tests must cover successful
captures and hostile arguments/environment/paths, timeout and output-limit failures, and install
rollback. Unprivileged tests do not establish that the real sudo policy works: verify it separately
after installation, including rejected invocations. Keep raw system-wide traces local.
