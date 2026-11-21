"""Run Python in a Windows Conda environment with a timeout and descendant cleanup.

Example: python scripts/windows_run.py --timeout 1800 --log runs/smoke.log --
         -m combat_rl.smoke --timeout-seconds 1800
Requires the project's train dependencies and environment-windows.yml prerequisites.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, required=True, help="Wall seconds")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("python_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    python_args = args.python_args[1:] if args.python_args[:1] == ["--"] else args.python_args
    if os.name != "nt" or args.timeout <= 0 or not python_args:
        parser.error("Requires Windows, a positive timeout, and Python arguments after --")
    prefix = Path(sys.executable).parent
    java_home = prefix / "Library/lib/jvm"
    if not (java_home / "bin/java.exe").is_file():
        parser.error("Run with Python from the Conda environment containing Java 21")
    env = os.environ.copy()
    env["CONDA_PREFIX"] = str(prefix)
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = os.pathsep.join(str(prefix / d) for d in
                                 ["Library/lib/jvm/bin", "Library/bin", "Scripts", ""]) + os.pathsep + env["PATH"]
    env["CMAKE_PREFIX_PATH"] = str(prefix / "Library")
    env["GRADLE_USER_HOME"] = str(prefix / "gradle-cache")
    env["PYTHONUNBUFFERED"] = env["PYTHONUTF8"] = "1"
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    known = {}
    code = 1
    command = [sys.executable, *python_args]
    # Exclusive creation preserves earlier diagnostics.
    with args.log.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        root = psutil.Process(process.pid)
        print(json.dumps({"pid": process.pid, "log": str(args.log), "timeout": args.timeout}), flush=True)
        try:
            while process.poll() is None:
                # Retain process identities so a reparented JVM can still be cleaned up.
                for parent in [root, *list(known.values())]:
                    try:
                        for child in parent.children(recursive=True):
                            known[(child.pid, child.create_time())] = child
                    except psutil.NoSuchProcess:
                        pass
                if time.monotonic() - started > args.timeout:
                    log.write("\nEXTERNAL TIMEOUT\n")
                    code = 124
                    break
                time.sleep(0.2)
            else:
                code = process.returncode
        finally:
            for child in reversed([root, *list(known.values())]):
                try:
                    if child.is_running():
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.wait(timeout=10)
            _, alive = psutil.wait_procs(list(known.values()), timeout=10)
    record = {"exit_code": code, "wall_seconds": time.monotonic() - started,
              "command": command, "remaining_descendant_pids": [p.pid for p in alive]}
    args.log.with_suffix(".status.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)
    return code if not alive else 1


if __name__ == "__main__":
    sys.exit(main())
