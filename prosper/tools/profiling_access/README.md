# Autonomous host scheduler profiling

One administrator installation grants one development account passwordless access to a
**scheduler-only capture helper**, not unrestricted `perf`, Python or a root shell. The client
then records from the host without prompting. This is intended for a dedicated development
machine where system-wide process/thread metadata may be collected for debugging.

CPU sampling and own-process debugger access often already work without this helper. It does
not grant unattended GDB attachment to other users, arbitrary tracing, CRIU, package management
or permission to change host settings.

## Install once (host terminal)

Review `capture.py` and `install.py` first. From the repository/worktree root, as your ordinary
user, replacing `YOUR_LOGIN` with that account's login:

```bash
sudo /usr/bin/python3 -I prosper/tools/profiling_access/install.py install --user YOUR_LOGIN
```

Host dependencies: Python 3, sudo/visudo, perf, and coreutils (`timeout`, `sleep`). Use the host,
not root inside a rootless Distrobox. On Fedora Atomic, `/usr/local` resolves to persistent
`/var/usrlocal`; the helper does not modify the image's `/usr/bin/perf`. No reboot is needed.

The installer creates exactly these managed files:

- `/usr/local/libexec/prosper-perf-capture`: root-owned executable copy, bound to the chosen UID.
- `/etc/sudoers.d/prosper-perf-capture`: root-owned rule authorizing only that executable.

It validates the existing and proposed sudoers policy, refuses existing files/symlinks, and
rolls back its own files if installation validation fails. The helper does **not** import
anything from the repository. Its isolated Python shebang ignores user Python paths/settings.
Editing the repository does not modify the installed privilege boundary; upgrades require
uninstall/reinstall after reviewing the new version. Protect the installed files and their
parent directories from non-root writes.

## Verify actual access (no sudo around the client)

```bash
python3 prosper/tools/profiling_access/profile.py --seconds 2 \
    --output prosper/build-access/first-capture --verify-access
```

The output directory must be new and on real disk shared with the build container. The client
uses `sudo -n`, so missing permission fails immediately instead of hanging at a password prompt.
Success requires actual switch and wakeup samples, then refusals of an excessive duration,
an injected command, and an output-path argument. A missing helper or denied sudo is not a
successful refusal control. Inspect `report.json`, `capture.err` and `events.err` for context.

This live check is separate from the unprivileged unit tests: mocks cannot certify sudo policy,
SELinux permission, kernel support or an installed helper. A successful event check does not
certify loss-free attribution to a particular game thread.

## Routine captures

Launch the game normally, establish the desired gameplay phase, then:

```bash
python3 prosper/tools/profiling_access/profile.py --seconds 15 \
    --output prosper/build-access/gameplay-run-01
perf sched timehist -i prosper/build-access/gameplay-run-01/scheduler.data -p "$target_pid"
```

Use the exact PID of the game you launched. Check for lost events and correct thread coverage
before comparing blocked time with runnable scheduling delay. Diagnostics perturb timing;
keep their runs separate from minimally instrumented performance comparisons.

For direct integration, the helper streams binary perf data on stdout:

```bash
sudo -n /usr/local/libexec/prosper-perf-capture scheduler 10 > "$run_dir/scheduler.data"
```

The **unprivileged shell** opens that file. Root receives no caller-selected output pathname.
This is perf's pipe format and can be decoded from the saved file using ordinary-user `perf`.
Use a new artifact path, restrict its permissions (the client does this automatically), and
check exit status: failed or interrupted captures may leave partial data. Raw traces can contain
other processes' names, addresses and paths; do not publish them.

## Fixed limits and security boundary

- Only `scheduler SECONDS`, with an integer duration of 1–60 seconds; no extra arguments.
- Only switch, wakeup, new-wakeup and migration tracepoints, recorded system-wide with period 1.
- A nonblocking root-owned lock permits only one helper capture at a time.
- Output is limited to 256 MiB, buffered in small chunks; blocked consumers have a deadline.
- Helper and recorder diagnostics are best-effort and nonblocking. A full/closed stderr pipe
  may lose diagnostics, but cannot keep a completed capture holding its lock. The helper restores
  inherited descriptor blocking flags on exit; its help output is bounded and best-effort too.
- An outer timeout bounds the recorder even if the supervising helper dies. Interrupts and
  supervisor failures clean up the recorder's process group. Kernel-uninterruptible tasks can
  still outlive userspace timeouts; this is not a kernel recovery mechanism.
- Root executes only trusted absolute-path system binaries with a fixed environment and cwd.
  User perf configuration, debuginfod lookup and build-ID caching are disabled.
- No global sysctl changes, tracefs permission changes, file capabilities, setuid installation,
  arbitrary workload launch, or shell command construction.

The account is intentionally allowed to observe scheduler activity across the machine, including
privileged tasks. This is a real observation privilege, even though it is not general root access.
Rate/size limits apply per invocation, not as a daily quota: the account can repeat captures and
consume its own disk space. Keep the host tools patched; a restricted interface does not eliminate
bugs in perf, Python, sudo, or the kernel.

Do not replace the rule with `NOPASSWD: /usr/bin/perf` or passwordless repository Python: perf can
launch arbitrary commands, and repository code is user-writable. See the
[sudoers security notes](https://man7.org/linux/man-pages/man5/sudoers.5.html),
[kernel perf privilege guidance](https://docs.kernel.org/admin-guide/perf-security.html), and
[Python isolated mode](https://docs.python.org/3/using/cmdline.html#cmdoption-I).

## Remove or upgrade (host administrator)

```bash
sudo /usr/bin/python3 -I prosper/tools/profiling_access/install.py uninstall
```

This removes the managed sudo rule first, then the helper. It refuses unrecognized/unsafe files.
Existing captures are preserved. A running capture may finish within its bound; removing policy
does not revoke an already-running process. The harmless root-owned `/run/prosper-perf-capture.lock`
may remain until reboot. Reinstall the reviewed version to upgrade or restore access.

## Local tests (unprivileged, Linux)

```bash
python3 prosper/tools/profiling_access/test_access.py
cmake -S prosper/tools/profiling_access -B prosper/build-access -G Ninja
ctest --test-dir prosper/build-access --no-tests=error --output-on-failure
```

The same test is registered in the main Linux CTest suite. These tests exercise policy, streaming
limits/cleanup, installer state transitions, and actual sudoers syntax parsing. They deliberately
do not install files or grant privileges on CI runners. Use the live access check after installation.
