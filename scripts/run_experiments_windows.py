"""Sequential Windows training queue with frozen settings and visible status.

Use only after selecting the common config on validation. Test seeds are not used.
Stop after the current run by creating RUN_ROOT/STOP_AFTER_CURRENT.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from combat_rl.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--conditions", choices=["A", "B", "C"], nargs="+", default=["A", "B", "C"])
    parser.add_argument("--timeout", type=int, default=21600, help="Per training process, seconds")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    if min(args.steps, args.timeout, args.episodes) < 1:
        parser.error("steps, timeout, and episodes must be positive")
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config["training"]["total_timesteps"] = args.steps
    frozen = root / "frozen-config.json"
    if frozen.exists() and json.loads(frozen.read_text()) != config:
        parser.error("Run root already has different settings; use a new root")
    frozen.write_text(json.dumps(config, indent=2) + "\n")
    source = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    state = {"source_revision": revision, "config": str(frozen), "completed": [],
             "steps_per_run": args.steps, "seeds": args.seeds, "conditions": args.conditions}

    def status(phase, **extra):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        temporary = root / "status.tmp.json"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "status.json")
        print(json.dumps(state), flush=True)

    def run(label, timeout, command):
        log = root / "logs" / (label + "-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".log")
        result = subprocess.run([sys.executable, str(source / "scripts/windows_run.py"),
                                 "--timeout", str(timeout), "--log", str(log), "--", *command], cwd=source)
        if result.returncode:
            raise RuntimeError(f"{label} exited {result.returncode}; inspect {log} and saved checkpoints")

    try:
        for seed in args.seeds:
            for condition in args.conditions:
                if (root / "STOP_AFTER_CURRENT").exists():
                    status("stopped_between_runs")
                    return 0
                name = f"{condition}_seed{seed}"
                directory = root / name
                # Refuse to silently mix prior models/configs. Resume a failed run
                # explicitly with combat_rl.train before restarting remaining jobs.
                if directory.exists() and any(directory.iterdir()):
                    raise RuntimeError(f"{directory} is not empty; choose remaining seeds/conditions or a new root")
                current_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
                dirty = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", "combat_rl", "scripts"], cwd=source, text=True).strip()
                if current_revision != revision or dirty:
                    raise RuntimeError("Source changed after queue start; stop and preserve experiment consistency")
                status("training", current_run=name)
                run(name+"-train", args.timeout, ["-m", "combat_rl.train", "--config", str(frozen),
                    "--condition", condition, "--seed", str(seed), "--steps", str(args.steps),
                    "--run-dir", str(directory), "--device", "cpu"])
                status("validation", current_run=name)
                run(name+"-validation", 3600, ["-m", "combat_rl.evaluate", "--model", str(directory/"final.zip"),
                    "--split", "validation", "--episodes", str(args.episodes), "--output", str(directory/"validation.csv")])
                state["completed"].append(name)
        status("complete", current_run=None)
        return 0
    except Exception as error:
        status("failed", error=str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
