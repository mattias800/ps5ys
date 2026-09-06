# Thread wait profiling — what blocks, how much, and where

**Status:** Linux implementation complete and merged (`tools/perf/wait_profile.py` #2377,
`tools/perf/stack_profile.py` #2380 + #2383, `tools/perf/lock_holder.py`). **Windows and macOS have
nothing.** This document exists so
the Windows lane can build the equivalent without re-deriving the design, and — more importantly — without
re-discovering the seven traps that made the Linux versions wrong on the way here.

Everything in §1–§4 is measured on Linux. **§5 (Windows) is a design sketch I could not test and must not
be trusted as verified** — it names which API answers which question and which properties must survive, not
a working recipe.

---

## 1. The question these answer, and why one tool cannot

*"Nothing is happening. What is everything waiting for?"*

That decomposes into two questions with different costs and different mechanisms, and **neither tool
subsumes the other**:

| Question | Tool | Mechanism | Cost |
| --- | --- | --- | --- |
| *How much* is each thread blocked, and on what object? | `wait_profile.py` | read `/proc` | negligible; sample at 10–50 Hz |
| *Where* in the code is it blocked? | `stack_profile.py` | attach gdb, walk stacks | stops the process; sample every 5–10 s |
| *Who HOLDS* the lock it is waiting on? | `lock_holder.py` | `process_vm_readv` of the mutex owner word | negligible; does **not** stop the target |

Run the cheap one first to find *which* thread is stuck, then the expensive one to find *where*.
"Thread A waits on lock X" is a symptom; "thread B holds X inside `foo()` every frame" is a fix — and
without the third question an investigation stops at a futex address, which names nothing.

### The limit that forces the pair, demonstrated

I built a control with three threads blocked in three known functions and read `/proc`:

```
tid=908640  state=S  wchan=futex_do_wait      syscall=202   <- blocked on a MUTEX
tid=908641  state=S  wchan=futex_do_wait      syscall=202   <- blocked on a CONDVAR
tid=908642  state=S  wchan=hrtimer_nanosleep  syscall=230
```

**A mutex wait and a condition-variable wait are identical in `/proc`** — same state, same `wchan`, same
syscall number. Anyone reading `wchan` to infer *which primitive* a thread is on is reading a field that
cannot carry that information. Only the stack separates them.

This is the single most important fact in this document. It is why the cheap tool is not enough, and it is
why a Windows implementation that stops at `WaitReason` will have the same blind spot.

---

## 2. `wait_profile.py` — how much, and on what

Reads three files per thread, none of which need privileges:

| File | Gives |
| --- | --- |
| `/proc/<pid>/task/<tid>/stat` | thread name, run state (`R`/`S`/`D`/…) |
| `/proc/<pid>/task/<tid>/wchan` | kernel function the thread is sleeping in |
| `/proc/<pid>/task/<tid>/syscall` | syscall number **and its arguments** |

**The load-bearing trick is syscall arg0.** For a futex wait that argument is the futex's *address*. On
Blue Prince that turned an unusable aggregate into an answer:

> main thread **54.4%** blocked on futexes `0xad2048` / `0xad204c` (four bytes apart), while 13
> `Job.Worker` threads parked on a **different** address.

Two threads on the same address are contending. Two on different addresses are not. The 13 workers are an
idle pool; the main thread is the problem. **A single "N% blocked" figure destroys exactly that
distinction**, which is the difference between a measurement and an answer.

### Design properties that must survive any port

- **Every sampled thread lands in exactly one bucket** (`RUNNING`, or `<state>:<wchan> [arg0]`), so shares
  sum to 100 by construction rather than by arithmetic that can drift.
- **Shares are per thread** — `n / (samples in which THAT thread was seen)` — so a thread that starts late
  is not diluted by samples it could not appear in.
- **The residual is printed, never absorbed.** With `--top` / `--min-share`, whatever is not displayed is
  shown as a remainder, so the column is visibly complete whatever the flags do.
- **The overran-sweep count prints even when zero**, so its absence is never ambiguous.

---

## 3. `stack_profile.py` — where

Attaches `gdb -p` with `set auto-solib-add off`, runs `thread apply all bt`, and reports the first
**application** frame beneath the wait machinery, aggregated per thread.

```
prosper-app/948563       30   53.3%  ioctl
AssetGarbageCol (x12)    30  100.0%  prosper::k_wait_on_address
Job.Worker 0..12         30  100.0%  prosper::interruptible_cond_wait <- prosper::k_ef_wait
GfxFlipThread            30  100.0%  std::condition_variable::wait_until
```

### It stops the process, so it says so first

~85–90 ms per sample on a 4-thread toy; **~0.6 s on prosper's 87 threads**. The report prints total stopped
time as a share of wall clock *before* any finding, and warns above 10%. At `--interval 3` that read 22.7%
and warned; at `--interval 8`, 8.3%. A profiler that quietly steals wall clock will manufacture the
frame-rate problem you came to investigate.

The figure is deliberately labelled an **upper bound** — it is gdb's whole wall time including spawn,
symbol work and detach, so the true stop is shorter. It errs high, which is the safe direction.

---

## 4. The seven traps. Read this section before writing any of it again

Each cost real time here. They are not Linux-specific — every one has a Windows analogue.

**1. An empty result is not a negative result.** gdb that cannot attach produces *no stacks*, which is
byte-identical to a process with nothing blocked. This fooled me during development: I ran with stderr
discarded, got an empty report, and read it as "nothing blocked" for a process with four blocked threads.
The tool now counts an empty sample as **FAILED**, prints the failed count **even when zero**, and exits
non-zero if every sample failed rather than printing a clean-looking empty report.

**2. Test debugger access in the target's environment.** The original measurements were:

```
process on host      + gdb on host       -> works
process in distrobox + gdb in distrobox  -> works
process on host      + gdb in distrobox  -> ptrace: Operation not permitted
```

These are observations of that setup, not universal host/container rules: credentials, namespaces
and security policy determine access. Prefer the target's environment, then require a successful
attach to the actual target. The [doctor child control](DEBUGGING_WORKFLOWS.md) checks only
same-environment child access. Per trap 1, discarding a denial can turn it into a clean-looking
empty profile; neither "always host" nor "always container" is a substitute for observing stacks.

**3. Versioned symbols defeat anchored patterns.** glibc reports `pthread_cond_wait@@GLIBC_2.3.2`. Any
`$`-anchored match over the raw string fails, so the wait wrapper is classified as *application* code and
**a libc internal gets reported as the blocking site** — a wrong answer that looks entirely plausible,
because it is a real function genuinely on the stack. Strip the version suffix before classifying.

**4. C++ wait wrappers are templated and no anchored pattern reaches them.**
`std::condition_variable::__wait_until_impl<std::chrono::duration<long, std::ratio<1l, 1000000000l> > >`
must be matched by **prefix**. Without this the tool reports `std::condition_variable::wait_until` as a
blocking *site* — naming the very machinery it exists to skip. `__gthread_cond_wait` is the trap within the
trap: a `_*` prefix eats the underscores, and then `pthread_cond_\w+` still needs a literal `pthread`.

**5. Unresolved (`??`) frames must never be the answer, and must not be silently elided.**
`auto-solib-add off` yields `??` for unloaded libraries. Two distinct mistakes:
   - `??` passing the plumbing filter and becoming the reported site (`?? <- ??`);
   - eliding `??` from the *middle* of a chain, so `["answer","??","caller2"]` renders `answer <- caller2`
     — asserting a "called by" adjacency that does not exist. Truncate and mark the gap instead.

**6. "No named site" has three causes and only one is a finding.** All frames resolved and all were waits →
a real statement about the program. Some frames unresolved → the tool could not name them. No frames parsed
→ a tool failure for that thread. Collapsing these into one label asserts a finding in the two cases that
support none. This is trap 1 applied per-thread, and it was missed *because* trap 1 had already been
"handled" per-sample.

**7. A control validates the classifier over the inputs it can generate, and is silent about inputs it
cannot.** My hand-built control had three threads blocked in three known functions and it correctly caught
trap 3. It ran on a box where gdb resolved libc and used raw `pthread_*` calls — so `??` frames and C++
wrapper frames were **structurally inexpressible in it**, and traps 4, 5 and 6 were invisible to it. All
three were found later by a reviewer *running* the classifier against cases the control could not build.
Build the control, and then assume it is blind to whole classes anyway.

---

## 5. Windows — design sketch, NOT verified

**I cannot test any of this.** It names the API that answers each question and the properties that must
survive. Treat every line as a starting point to verify, not a result.

### 5a. The `wait_profile` equivalent

There is no `/proc`. The closest single source is
**`NtQuerySystemInformation(SystemProcessInformation)`**, which returns `SYSTEM_THREAD_INFORMATION` per
thread including:

| Field | Linux analogue |
| --- | --- |
| `ThreadState` | `stat` field 3 (`R`/`S`/`D`) |
| `WaitReason` (`KWAIT_REASON`) | `wchan`, roughly |
| `ClientId.UniqueThread` | tid |

**The hard part is the futex-address analogue, and it is the whole value of the Linux tool.** `WaitReason`
alone is the aggregate that cannot distinguish contention from an idle pool. Candidates, none verified:

- For **SRW locks / critical sections**, the object address is not exposed by
  `NtQuerySystemInformation`. WinDbg's `!locks` and `!cs` walk them; whether an equivalent is reachable
  in-process without a debugger attach is the open question.
- **`GetThreadWaitChain`** (Wait Chain Traversal, `advapi32`) is the most promising: it is *designed* to
  answer "who is waiting on what, and who holds it", returning `WAITCHAIN_NODE_INFO` with object type and
  thread/lock identity. It may deliver more than the Linux tool does, since it resolves the *holder* and
  not merely a shared address.
- If neither works, the stack sampler subsumes this question on Windows and the two-tool split may not be
  the right decomposition there. **That is a legitimate outcome** — the split exists because `/proc` is
  cheap and stacks are expensive, and if Windows has no cheap layer, do not invent one for symmetry.

### 5b. The `stack_profile` equivalent

- **`cdb.exe` / `ntsd`** with `~*k` is the direct analogue of `gdb -p -ex 'thread apply all bt'` and keeps
  the same shape: attach, dump all thread stacks, detach, aggregate. Same perturbation obligation — it
  stops the process, so report the stopped share **before** any finding.
- **In-process** `StackWalk64`/`dbghelp` avoids the attach entirely but only walks the *calling* thread
  cheaply; suspending peers to walk them reintroduces the stop.
- The classifier (traps 3–6) is **platform-independent in shape but not in content**: the plumbing set
  needs the MSVC/Windows spellings — `RtlUserThreadStart`, `NtWaitForSingleObject`,
  `WaitForSingleObjectEx`, `SleepConditionVariableSRW`, `RtlSleepConditionVariableSRW`,
  `AcquireSRWLockExclusive`, and the MSVC STL's `_Thrd_sleep` / `_Cnd_wait` family.
  `prosper/tools/perf/test_stack_profile.py` shows the test shape: classify **recorded** debugger output so
  the tests need no debugger and cannot skip silently.

### 5c. What must survive the port, whatever the mechanism

These are the properties, not the implementation:

1. An empty or failed sample is **reported as failed**, never as "nothing was blocked", and an all-failed
   run exits non-zero.
2. Perturbation is measured and printed **before** any finding, labelled as a bound with its direction.
3. Buckets partition by construction; shares are per thread; the residual is printed, not absorbed.
4. Wait machinery is never reported as a blocking site — and the plumbing list is validated by a
   **paired negative** proving real application frames are not swallowed. A filter that classifies
   everything as plumbing passes every other test you can write for it.
5. Tests classify **recorded** output, so they cannot skip silently where the debugger is unavailable.

---

## 6. What these tools have actually produced

- **#2387 / #2389** — `guest_writable` re-parsing `/proc/self/maps` on every call on Linux, found from the
  main thread's stdio frames (`_IO_default_uflow`, `__vfscanf_internal`, `fgets`). **Windows and macOS
  answer that predicate with a syscall per region, so that defect is invisible from the Windows lane** —
  a concrete instance of why per-platform profiling coverage matters.
- **#2381** — a render-target readback hypothesis of mine, **falsified**: it measured 25–50% at n=8/n=12
  and then appeared in **zero of 46** further samples, with the renderer's own counters agreeing
  (`persist_miss=0.0`, `texture_cache misses=19`). Recorded because the tools killing a wrong answer early
  is the point of having them.

**Neither has yet explained Blue Prince's frame rate.** The title still runs at roughly 1 fps in ordinary
play. Nothing in this document should be read as a frame-rate fix; these tools narrow where to look, and
the looking is not finished.
