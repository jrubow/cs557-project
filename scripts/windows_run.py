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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.queue_optimization_windows import write_json


def supervise(command, env, log_path, timeout):
    """Always persist observed child/cleanup outcomes, including supervisor errors."""
    started, known, errors = time.monotonic(), {}, []
    process, root, code, child_code = None, None, 1, None
    with log_path.open('x', encoding='utf-8') as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, creationflags=subprocess.CREATE_NO_WINDOW)
            root = psutil.Process(process.pid)
            while process.poll() is None:
                for parent in [root, *list(known.values())]:
                    try:
                        for child in parent.children(recursive=True):
                            known[(child.pid, child.create_time())] = child
                    except psutil.NoSuchProcess:
                        pass
                if time.monotonic()-started > timeout:
                    log.write('\nEXTERNAL TIMEOUT\n')
                    code = 124
                    break
                time.sleep(.2)
            else:
                code = child_code = process.returncode
        except (Exception, KeyboardInterrupt) as error:
            errors.append(repr(error))
        finally:
            for child in reversed(([root] if root is not None else [])+list(known.values())):
                try:
                    if child.is_running():
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
                except Exception as error:
                    errors.append(repr(error))
            if process is not None:
                try:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=10)
                except Exception as error:
                    errors.append(repr(error))
            try:
                _, alive = psutil.wait_procs(list(known.values()), timeout=10)
            except Exception as error:
                errors.append(repr(error))
                alive = list(known.values())  # Unknown cleanup cannot be called clean.
            if errors:
                log.write('\nSUPERVISOR ERRORS: '+json.dumps(errors)+'\n')
                if code == 0:
                    code = 1
    record = dict(exit_code=code, child_exit_code=child_code,
                  wall_seconds=time.monotonic()-started, command=command,
                  remaining_descendant_pids=[p.pid for p in alive], supervisor_errors=errors)
    write_json(log_path.with_suffix('.status.json'), record)
    return record


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
    env['CS557_SUPERVISED_WINDOWS'] = '1'
    args.log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, *python_args]
    record = supervise(command, env, args.log, args.timeout)
    print(json.dumps(record), flush=True)
    return record['exit_code'] if not record['remaining_descendant_pids'] else 1


if __name__ == "__main__":
    sys.exit(main())
