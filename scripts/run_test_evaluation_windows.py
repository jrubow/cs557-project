"""Evaluate a locked comparison and optimized recipe on held-out scenarios.

This queue only evaluates; it contains no training or model-selection step. Each
PPO model and both full/restricted reference controllers receive the same 50 test
scenario seeds. Run once into a new output root and preserve failures for review.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from queue_optimization_windows import clean_supervisor, comparison_results, inspect_model, read_json, sha256, write_json


METRICS = ("waves_completed", "success", "kills", "damage_dealt", "damage_taken", "elapsed_seconds")


def git(source, *args):
    return subprocess.check_output(["git", *args], cwd=source, text=True).strip()


def plan(args):
    comparison, optimization = args.comparison_root.resolve(), args.optimization_root.resolve()
    source = args.source.resolve()
    if read_json(comparison / "status.json")["phase"] != "complete" or read_json(optimization / "status.json")["phase"] != "complete":
        raise RuntimeError("Both previous queues must complete before test evaluation")
    if git(source, "status", "--porcelain"):
        raise RuntimeError("Evaluation source must be a clean frozen checkout")
    revision = git(source, "rev-parse", "HEAD")
    if revision != read_json(optimization / "manifest.json")["source_revision"]:
        raise RuntimeError("Use the frozen optimization checkout for evaluation")
    for path in (optimization / "logs").glob("*.status.json"):
        clean_supervisor(path)
    selected = read_json(optimization / "final-selection.json")
    if selected["status"] != "validated_improvement" or selected["test_split_used"]:
        raise RuntimeError("Expected the validation-selected recipe before its first test evaluation")
    comparison_hash = sha256(comparison / "manifest.json")
    manifest, results, _ = comparison_results(comparison, comparison_hash)
    config = manifest["config"]
    if config["evaluation"] != {"episodes": 50, "validation_seed": 50000, "test_seed": 100000}:
        raise RuntimeError("Unexpected evaluation split configuration")
    jobs = []
    for condition, records in results.items():
        for record in records:
            jobs.append(dict(name=f"{condition}_1M_seed{record['seed']}", group=f"PPO {condition} 1M",
                             policy="ppo", condition=condition, training_seed=record["seed"],
                             training_target=1000000, **{k:record[k] for k in ("model", "model_sha256", "metadata_sha256", "steps")}))
    if len(selected["models"]) != 3 or selected["recipe"]["target"] != 2000000:
        raise RuntimeError("Expected three selected 2M checkpoints")
    for seed, model in enumerate(selected["models"]):
        identity = inspect_model(model, config, selected["condition"], seed, 2000000, revision)
        jobs.append(dict(name=f"{selected['condition']}_2M_seed{seed}", group=f"PPO {selected['condition']} 2M",
                         policy="ppo", condition=selected["condition"], training_seed=seed,
                         training_target=2000000, **identity))
    # A and B have identical full observations and task dynamics; only reward
    # differs. Controllers do not consume rewards, so B is the full-view reference.
    for policy in ("random", "heuristic"):
        for condition, view in (("B", "full"), ("C", "restricted")):
            jobs.append(dict(name=f"{policy}_{view}", group=f"{policy} {view}", policy=policy,
                             condition=condition, training_seed="", training_target=0))
    return dict(source=str(source), source_revision=revision, config=config, jobs=jobs,
                episodes_per_job=50, first_test_seed=100000, last_test_seed=100049,
                deterministic_ppo=True, selection_frozen_before_test=True,
                optimization_selection=str(optimization / "final-selection.json"),
                selection_sha256=sha256(optimization / "final-selection.json"),
                comparison_manifest=str(comparison / "manifest.json"), comparison_manifest_sha256=comparison_hash,
                protocol_sha256=sha256(__file__),
                note="Same scenario seed list does not make Minecraft AI bit-exact. No test-driven model changes. "
                     "Optimized C uses twice the training budget and is reported separately. "
                     "Full-view references apply to A/B; reward values are not compared across conditions.")


def check_identity(job):
    if job["policy"] == "ppo":
        model = Path(job["model"])
        if sha256(model) != job["model_sha256"] or sha256(model.with_suffix(".json")) != job["metadata_sha256"]:
            raise RuntimeError(f"Locked model changed: {job['name']}")


def validate_result(path, job, config):
    meta = read_json(path.with_suffix(".json"))
    expected_model = Path(job["model"]).resolve() if job["policy"] == "ppo" else None
    if (meta["split"] != "test" or meta["first_seed"] != 100000 or meta["episodes"] != 50
            or meta["condition"] != job["condition"] or meta["policy"] != job["policy"]
            or str(meta["training_seed"]) != str(job["training_seed"]) or meta["config"] != config
            or not meta["deterministic"]
            or (expected_model is not None and Path(meta["model"]).resolve() != expected_model)):
        raise RuntimeError(f"Test metadata mismatch: {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 50 or [int(row["scenario_seed"]) for row in rows] != list(range(100000, 100050)):
        raise RuntimeError(f"Wrong, duplicated, or missing test scenarios: {path}")
    for row in rows:
        if (row["split"] != "test" or row["condition"] != job["condition"] or row["policy"] != job["policy"]
                or row["training_seed"] != str(job["training_seed"]) or int(row["wave_target"]) != 5
                or row["success"] not in ("True", "False")
                or row["terminal_reason"] not in ("death", "success", "time_limit")
                or (expected_model is not None and Path(row["model"]).resolve() != expected_model)):
            raise RuntimeError(f"Invalid test row identity: {path}")
        for key in METRICS:
            value = float(row[key]) if key != "success" else float(row[key] == "True")
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"Invalid test metric: {key}")
        if not 0 <= float(row["waves_completed"]) <= 5 or (row["success"] == "True") != (float(row["waves_completed"]) == 5):
            raise RuntimeError("Success and wave counters disagree")
    record = dict(name=job["name"], group=job["group"], policy=job["policy"], condition=job["condition"],
                  training_seed=job["training_seed"], training_target=job["training_target"], episodes=50,
                  test_csv=str(path), successes=sum(row["success"] == "True" for row in rows))
    for key in METRICS:
        record[key] = statistics.mean(float(r[key]) if key != "success" else float(r[key] == "True") for r in rows)
    record["terminal_reasons"] = {reason: sum(r["terminal_reason"] == reason for r in rows)
                                  for reason in ("death", "success", "time_limit")}
    return record


def summarize(root, records):
    groups = []
    for group in dict.fromkeys(r["group"] for r in records):
        runs = [r for r in records if r["group"] == group]
        item = dict(group=group, independent_training_runs=len(runs) if runs[0]["policy"] == "ppo" else 0,
                    evaluation_runs=len(runs), episodes=sum(r["episodes"] for r in runs),
                    successes=sum(r["successes"] for r in runs))
        for key in METRICS:
            values = [r[key] for r in runs]
            item[key + "_mean"] = statistics.mean(values)
            item[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        groups.append(item)
    write_json(root / "results.json", dict(split="test", first_seed=100000, last_seed=100049,
               runs=records, groups=groups, complete=len(records) == 16,
               note="Means weight training seeds equally. SD describes variation across three trained models; "
                    "episodes are not independent training runs. References have one 50-scenario evaluation each."))
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=groups[0])
        writer.writeheader()
        writer.writerows(groups)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--optimization-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol = plan(args)
    if args.check_only:
        print(json.dumps(dict(jobs=len(protocol["jobs"]), episodes=50 * len(protocol["jobs"]),
                             source_revision=protocol["source_revision"], selection_sha256=protocol["selection_sha256"])))
        return 0
    root, source = args.output_root.resolve(), Path(protocol["source"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("x") as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), created=psutil.Process().create_time())))
    if (root / "manifest.json").exists():
        raise RuntimeError("Test manifest already exists; do not repeat or overwrite test results")
    if any(c.status == "LISTEN" and c.laddr.port == 8000 for c in psutil.net_connections(kind="tcp") if c.laddr):
        raise RuntimeError("Simulator port is occupied")
    write_json(root / "manifest.json", dict(protocol, frozen_utc=datetime.now(timezone.utc).isoformat()))
    shutil.copy2(protocol["optimization_selection"], root / "selection-before-test.json")
    shutil.copy2(__file__, root / "protocol.py")
    write_json(root / "baseline-config.json", protocol["config"])
    state = dict(completed=[], total_jobs=len(protocol["jobs"]), total_episodes=800, split="test")
    def status(phase, **extra):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(root / "status.json", state)
        print(json.dumps(state), flush=True)
    results = []
    start = time.monotonic()
    status("starting")
    try:
        for job in protocol["jobs"]:
            if (root / "STOP_AFTER_CURRENT").exists():
                status("stopped", current_job=None)
                return 0
            if (git(source, "rev-parse", "HEAD") != protocol["source_revision"] or git(source, "status", "--porcelain")
                    or sha256(__file__) != protocol["protocol_sha256"]
                    or sha256(protocol["optimization_selection"]) != protocol["selection_sha256"]
                    or sha256(protocol["comparison_manifest"]) != protocol["comparison_manifest_sha256"]):
                raise RuntimeError("Locked protocol/source/selection changed")
            check_identity(job)
            output = root / "evaluations" / (job["name"] + ".csv")
            command = ["-m", "combat_rl.evaluate", "--policy", job["policy"], "--split", "test",
                       "--episodes", "50", "--output", str(output)]
            if job["policy"] == "ppo":
                command += ["--model", job["model"]]
                config = read_json(Path(job["model"]).with_suffix(".json"))["config"]
            else:
                command += ["--condition", job["condition"], "--config", str(root / "baseline-config.json")]
                config = protocol["config"]
            status("evaluating", current_job=job["name"])
            log = root / "logs" / (job["name"] + ".log")
            process = subprocess.run([sys.executable, str(source / "scripts/windows_run.py"), "--timeout", "3600",
                                      "--log", str(log), "--", *command], cwd=source)
            supervised = clean_supervisor(log.with_suffix(".status.json"))
            if process.returncode:
                raise RuntimeError(f"Test supervisor exited {process.returncode}")
            check_identity(job)
            result = validate_result(output, job, config)
            result.update(wall_seconds=supervised["wall_seconds"], csv_sha256=sha256(output))
            results.append(result)
            state["completed"].append(job["name"])
            summarize(root, results)
        for job in protocol["jobs"]:
            check_identity(job)
        status("complete", current_job=None, wall_seconds=time.monotonic() - start, models_unchanged=True)
        return 0
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    sys.exit(main())
