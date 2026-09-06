#!/usr/bin/python3 -I
"""Administrator-only installation/removal of the restricted profiling helper.

Usage: sudo /usr/bin/python3 -I install.py install --user LOGIN
       sudo /usr/bin/python3 -I install.py uninstall
Refuses existing installations; uninstall first to upgrade. Never edits global perf restrictions.
"""
import argparse
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
import sys
import tempfile

HELPER = Path("/usr/local/libexec/prosper-perf-capture")
RULE = Path("/etc/sudoers.d/prosper-perf-capture")
MARKER = b"# prosper-perf-managed-v1\n"
ENV = {"PATH": "/usr/bin:/usr/sbin", "LC_ALL": "C"}


def protected_directory(path):
    resolved = path.resolve(strict=True)
    for item in [resolved, *resolved.parents]:
        info = item.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise PermissionError(f"untrusted installation directory: {item}")


def validate_username(username, lookup=pwd.getpwnam):
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", username):
        raise ValueError("unsupported login spelling; use a simple existing local account name")
    uid = lookup(username).pw_uid
    if uid <= 0:
        raise ValueError("choose a non-root development account")
    return uid


def helper_payload(source, uid):
    text = source.read_text()
    token = "ALLOWED_UID = None  # installer replaces this"
    if not text.startswith("#!/usr/bin/python3 -I\n") or text.count(token) != 1 or MARKER.decode() not in text:
        raise ValueError("unexpected helper source; refusing installation")
    text = text.replace(token, f"ALLOWED_UID = {uid}  # installed account")
    compile(text, "installed-prosper-perf-capture", "exec")
    return text.encode()


def sudoers_payload(username, helper=HELPER):
    # No wildcard in the executable path; argument validation belongs to the fixed helper.
    return MARKER + f"{username} ALL=(root) NOPASSWD: NOSETENV: {helper}\n".encode()


def stage(path, data, mode):
    fd, name = tempfile.mkstemp(prefix=".prosper-perf-", dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        return staged
    except BaseException:
        staged.unlink()
        raise


def install(username, uid, source, visudo, helper=HELPER, rule=RULE):
    if os.path.lexists(helper) or os.path.lexists(rule):
        raise FileExistsError("helper or sudo rule already exists; inspect it and uninstall before upgrading")
    protected_directory(rule.parent)
    protected_directory(helper.parent.parent)
    helper.parent.mkdir(mode=0o755, exist_ok=True)
    protected_directory(helper.parent)
    payload = helper_payload(source, uid)
    # Both global policy and the proposed isolated rule must parse before enabling access.
    subprocess.run([visudo, "-c"], check=True, env=ENV, cwd="/", stdin=subprocess.DEVNULL)
    staged = []
    published = []
    try:
        h = stage(helper, payload, 0o755)
        staged.append(h)
        r = stage(rule, sudoers_payload(username, helper), 0o440)
        staged.append(r)
        subprocess.run([visudo, "-c", "-f", str(r)], check=True, env=ENV, cwd="/",
                       stdin=subprocess.DEVNULL)
        for candidate, destination in ((h, helper), (r, rule)):
            # Atomic creation, never replacement of a concurrent/existing file.
            os.link(candidate, destination)
            published.append(destination)
        subprocess.run([visudo, "-c"], check=True, env=ENV, cwd="/", stdin=subprocess.DEVNULL)
    except BaseException:
        # Revoke policy first; only files created by this invocation are removed.
        for destination in reversed(published):
            destination.unlink()
        raise
    finally:
        for candidate in staged:
            candidate.unlink()


def managed_file(path):
    if not os.path.lexists(path):
        return False
    protected_directory(path.parent)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022
            or info.st_nlink != 1 or MARKER not in path.read_bytes()[:256]):
        raise PermissionError(f"not a recognized protected helper installation: {path}")
    return True


def uninstall(helper=HELPER, rule=RULE):
    # Validate both before removing either. A missing component permits recovery from a partial install.
    existing = [path for path in (rule, helper) if managed_file(path)]
    for path in existing:
        path.unlink()
    return existing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("install", "uninstall"))
    parser.add_argument("--user", help="existing local non-root login to authorize")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("run this installer once from a host administrator terminal")
    if (args.action == "install") != bool(args.user):
        parser.error("--user is required only for install")
    try:
        if args.action == "install":
            uid = validate_username(args.user)
            # Check trusted host binaries without importing code from the working directory.
            visudo = "/usr/sbin/visudo" if Path("/usr/sbin/visudo").is_file() else "/usr/bin/visudo"
            for name in (visudo, "/usr/bin/python3", "/usr/bin/perf", "/usr/bin/timeout", "/usr/bin/sleep"):
                binary = Path(name).resolve(strict=True)
                protected_directory(binary.parent)
                st = binary.stat()
                if st.st_uid != 0 or st.st_mode & 0o022 or not stat.S_ISREG(st.st_mode) or not st.st_mode & 0o111:
                    raise PermissionError(f"untrusted host executable: {name}")
            install(args.user, uid, Path(__file__).with_name("capture.py"), visudo)
            print(f"Installed scheduler-only access for {args.user}: {HELPER}")
            print("Next: run the unprivileged verification command in README.md; no reboot is needed.")
        else:
            removed = uninstall()
            print("Removed: " + (", ".join(str(p) for p in removed) or "nothing (not installed)"))
            print("Access revoked; captures and the harmless /run lock are retained. Reinstall to restore access.")
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Installation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
