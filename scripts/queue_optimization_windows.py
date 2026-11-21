"""Bounded validation-only optimization after a supervised nine-run comparison.

The waiting process starts no simulator until its predecessor exits successfully.
Use a clean, dedicated checkout and a new output root. STOP_AFTER_CURRENT stops
between supervised jobs; outputs are never reused automatically after failure.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile

import psutil


SEEDS = (0, 1, 2)
CONDITIONS = ("A", "B", "C")
METRICS = ("success_rate", "waves", "kills", "damage_dealt", "damage_taken")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def average(scores):
    return {key: statistics.mean(score[key] for score in scores) for key in METRICS}


def rank(score):
    return tuple(score[key] for key in ("success_rate", "waves", "kills"))


def qualifies(candidate, baseline):
    success = candidate["success_rate"] - baseline["success_rate"]
    waves = candidate["waves"] - baseline["waves"]
    return success >= 0.06 - 1e-9 or (success >= -1e-9 and waves >= 0.25 - 1e-9)


def confirmed(candidates, baselines):
    current, previous = average(candidates), average(baselines)
    if not qualifies(current, previous) or current["success_rate"] < previous["success_rate"] - 1e-9:
        return False
    metric = "success_rate" if current["success_rate"] - previous["success_rate"] >= 0.06 - 1e-9 else "waves"
    return sum(c[metric] > b[metric] + 1e-9 for c, b in zip(candidates, baselines)) >= 2


def clean_supervisor(path):
    result = read_json(path)
    if result["exit_code"] != 0 or result["remaining_descendant_pids"]:
        raise RuntimeError(f"Failed or unclean supervised job: {path}")
    return result


def inspect_model(model, config, condition, seed, target, revision, fresh=False):
    model = Path(model)
    metadata = read_json(model.with_suffix(".json"))
    expected = copy.deepcopy(config)
    expected["training"]["total_timesteps"] = target
    rollout = expected["training"]["n_steps"] * expected["training"]["n_envs"]
    if (metadata["config"] != expected or metadata["condition"] != condition
            or metadata["seed"] != seed or not target <= metadata["timesteps"] < target + rollout
            or metadata["runtime"]["git_revision"] != revision or metadata["runtime"]["git_status"]):
        raise RuntimeError(f"Checkpoint provenance/budget mismatch: {model}")
    if fresh and metadata["resume_from"] is not None:
        raise RuntimeError(f"Expected fresh training: {model}")
    with zipfile.ZipFile(model) as archive:
        if json.loads(archive.read("data"))["num_timesteps"] != metadata["timesteps"]:
            raise RuntimeError(f"Model/metadata step mismatch: {model}")
    return {"model": str(model.resolve()), "model_sha256": sha256(model),
            "metadata_sha256": sha256(model.with_suffix(".json")), "steps": metadata["timesteps"]}


def evaluation_score(path, model, config, condition, seed, first_seed=None):
    path, model = Path(path), Path(model)
    metadata = read_json(path.with_suffix(".json"))
    first = config["evaluation"]["validation_seed"] if first_seed is None else first_seed
    if not config["evaluation"]["validation_seed"] <= first < first + 50 <= config["evaluation"]["test_seed"]:
        raise RuntimeError("Validation range crosses split boundary")
    if (metadata["split"] != "validation" or metadata["episodes"] != 50
            or metadata["first_seed"] != first or metadata["condition"] != condition
            or metadata["training_seed"] != seed or metadata["config"] != config
            or not metadata["deterministic"] or Path(metadata["model"]).resolve() != model.resolve()):
        raise RuntimeError(f"Validation metadata mismatch: {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 50 or [int(row["scenario_seed"]) for row in rows] != list(range(first, first + 50)):
        raise RuntimeError(f"Missing, duplicated, or unexpected validation scenarios: {path}")
    scores = []
    for row in rows:
        if (row["policy"] != "ppo" or row["split"] != "validation" or row["condition"] != condition
                or int(row["training_seed"]) != seed or int(row["wave_target"]) != 5
                or Path(row["model"]).resolve() != model.resolve()
                or row["success"] not in ("True", "False")
                or row["terminal_reason"] not in ("success", "death", "time_limit")):
            raise RuntimeError(f"Invalid validation row: {path}")
        score = dict(success_rate=float(row["success"] == "True"), waves=float(row["waves_completed"]),
                     kills=float(row["kills"]), damage_dealt=float(row["damage_dealt"]),
                     damage_taken=float(row["damage_taken"]))
        if (not all(math.isfinite(value) and value >= 0 for value in score.values())
                or not 0 <= score["waves"] <= 5
                or bool(score["success_rate"]) != (score["waves"] == 5)):
            raise RuntimeError(f"Invalid validation metrics: {path}")
        scores.append(score)
    return average(scores)


def comparison_results(root, expected_hash):
    root = Path(root)
    if sha256(root / "manifest.json") != expected_hash:
        raise RuntimeError("Comparison manifest changed")
    manifest = read_json(root / "manifest.json")
    config = manifest["config"]
    if manifest["steps_per_run"] != 1_000_000 or manifest["validation_episodes"] != 50:
        raise RuntimeError("Expected the frozen 1M/50-episode comparison")
    results, durations = {}, []
    for condition in CONDITIONS:
        results[condition] = []
        for seed in SEEDS:
            name = f"{condition}_seed{seed}"
            for stage in ("train", "validation"):
                paths = list((root / "logs").glob(f"{name}-{stage}-*.status.json"))
                if len(paths) != 1:
                    raise RuntimeError(f"Expected one successful {stage} supervisor for {name}")
                supervised = clean_supervisor(paths[0])
                if stage == "train":
                    durations.append(supervised["wall_seconds"])
            model = root / name / "final.zip"
            identity = inspect_model(model, config, condition, seed, 1_000_000,
                                     manifest["source_revision"], fresh=True)
            score = evaluation_score(root / name / "validation.csv", model, config, condition, seed)
            results[condition].append(dict(identity, seed=seed, score=score))
    return manifest, results, durations


def diagnostic_summary(trace, progress, config):
    with Path(trace).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Empty behavioral trace")
    episode_path = Path(str(trace).replace(".trace.csv", ".csv"))
    with episode_path.open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    by_scenario = {}
    for row in rows:
        by_scenario.setdefault(row["scenario_seed"], []).append(row)
    if set(by_scenario) != {row["scenario_seed"] for row in episodes}:
        raise RuntimeError("Trace/episode scenario mismatch")
    for episode in episodes:
        decisions = by_scenario[episode["scenario_seed"]]
        if [int(r["decision"]) for r in decisions] != list(range(int(episode["elapsed_steps"]))):
            raise RuntimeError("Trace/episode decision count mismatch")
        for trace_key, episode_key in (("reward", "episode_return"), ("kills_delta", "kills"),
                                       ("damage_dealt_delta", "damage_dealt"), ("damage_taken_delta", "damage_taken")):
            if not math.isclose(sum(float(r[trace_key]) for r in decisions), float(episode[episode_key]), abs_tol=1e-5):
                raise RuntimeError(f"Trace/episode {episode_key} mismatch")
    wave_health = {}
    first_seen = set()
    for row in rows:
        key = (row["scenario_seed"], row["wave"])
        if key not in first_seen:
            first_seen.add(key)
            wave_health.setdefault(row["wave"], []).append(float(row["health"]))
    radius = config["environment"]["arena_radius"]
    close = [r for r in rows if r["nearest_distance"] and float(r["nearest_distance"]) <= 3]
    attacking = [r for r in rows if int(r["attack"])]
    result = {
        "decisions": len(rows), "fraction_within_3_blocks": len(close) / len(rows),
        "fraction_near_wall": sum(max(abs(float(r["x"])), abs(float(r["z"]))) > radius - 2 for r in rows) / len(rows),
        "fraction_stationary_requested": sum(int(r["move"]) == int(r["strafe"]) == 0 for r in rows) / len(rows),
        "fraction_attacking": len(attacking) / len(rows),
        "fraction_attack_decisions_without_damage": (sum(float(r["damage_dealt_delta"]) == 0 for r in attacking)
                                                     / len(attacking) if attacking else None),
        "mean_absolute_bearing_when_close": statistics.mean(abs(float(r["nearest_bearing"])) for r in close) if close else None,
        "mean_health_entering_wave": {wave: statistics.mean(health) for wave, health in wave_health.items()},
        "note": "Attack/no-damage and wall proximity are diagnostics, not proof of cooldown or pathing failure.",
    }
    with Path(progress).open(newline="") as stream:
        updates = list(csv.DictReader(stream))[-100:]
    for column in ("train/approx_kl", "train/clip_fraction", "train/entropy_loss", "train/explained_variance"):
        values = [float(r[column]) for r in updates if r.get(column) and math.isfinite(float(r[column]))]
        if not values:
            raise RuntimeError(f"Missing diagnostic training metric: {column}")
        result[column] = statistics.median(values)
    result["normalized_joint_entropy"] = -result["train/entropy_loss"] / math.log(3 * 3 * 5 * 3 * 2 * 2 * 2)
    return result


def proposed_settings(diagnostics):
    candidates = []
    if diagnostics["train/approx_kl"] >= 0.03 or diagnostics["train/clip_fraction"] >= 0.30:
        candidates.append(dict(name="lower_lr", changes={"learning_rate": 0.0001},
                               hypothesis="Large recent PPO updates may limit stable improvement."))
    entropy = diagnostics["normalized_joint_entropy"]
    if entropy >= 0.60:
        candidates.append(dict(name="lower_entropy", changes={"ent_coef": 0.005},
                               hypothesis="Persistent action noise may limit precise combat behavior."))
    elif entropy <= 0.25:
        candidates.append(dict(name="higher_entropy", changes={"ent_coef": 0.02},
                               hypothesis="Early action collapse may limit exploration of useful behavior."))
    return candidates


class BudgetStop(Exception):
    pass


class Queue:
    def __init__(self, args):
        self.args = args
        self.root = args.output_root.resolve()
        self.live = args.comparison_root.resolve()
        self.source = Path(__file__).resolve().parents[1]
        self.revision = self.git("rev-parse", "HEAD")
        self.active_start = None
        self.state = {"source_revision": self.revision, "test_split_used": False,
                      "comparison_root": str(self.live), "hours_cap": args.hours,
                      "cutoff_utc": args.cutoff_utc.isoformat(), "jobs_completed": 0}
        self.screen_reserve = 6 * 3600
        self.rate = 0.01
        self.selection = None

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.source, text=True).strip()

    def assert_source(self):
        if self.git("rev-parse", "HEAD") != self.revision or self.git("status", "--porcelain"):
            raise RuntimeError("Optimization checkout changed; preserve the pinned experiment source")

    def status(self, phase, **extra):
        old = self.state.get("phase")
        self.state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        if self.active_start is not None:
            self.state["active_wall_seconds"] = time.monotonic() - self.active_start
        write_json(self.root / "status.json", self.state)
        if phase != old or extra:
            print(json.dumps(self.state), flush=True)

    def ledger(self, **entry):
        with (self.root / "ledger.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), **entry), allow_nan=False) + "\n")

    def remaining(self):
        deadline_left = (self.args.cutoff_utc - datetime.now(timezone.utc)).total_seconds()
        compute_left = self.args.hours * 3600 - (time.monotonic() - self.active_start if self.active_start is not None else 0)
        return min(deadline_left, compute_left)

    def check_stop(self):
        if (self.root / "STOP_AFTER_CURRENT").exists() or (self.live / "STOP_AFTER_CURRENT").exists():
            raise BudgetStop("Stop marker found; no further simulator job started")
        if self.remaining() <= 30:
            raise BudgetStop("Compute/deadline limit reached")

    def predecessor_alive(self):
        try:
            p = psutil.Process(self.args.comparison_pid)
            return abs(p.create_time() - self.args.comparison_created) < 0.01 and p.is_running()
        except psutil.NoSuchProcess:
            return False

    def wait_for_comparison(self):
        deadline = time.monotonic() + self.args.wait_hours * 3600
        expected = {f"{c}_seed{s}" for s in (0, 2) for c in CONDITIONS}
        self.status("waiting_for_comparison", predecessor_pid=self.args.comparison_pid)
        while True:
            self.check_stop()
            if sha256(self.live / "manifest.json") != self.args.comparison_manifest_sha256:
                raise RuntimeError("Comparison manifest changed while waiting")
            state = read_json(self.live / "status.json")
            alive = self.predecessor_alive()
            if state["phase"] == "complete" and set(state["completed"]) == expected and not alive:
                return
            if state["phase"] not in ("training", "validation", "complete"):
                raise RuntimeError(f"Comparison stopped in phase {state['phase']}; optimization not started")
            if not alive:
                raise RuntimeError("Comparison process exited without the expected successful batch completion")
            if time.monotonic() >= deadline:
                raise BudgetStop("Timed out waiting for comparison; no optimization simulator started")
            self.status("waiting_for_comparison")
            time.sleep(15)

    def run_job(self, label, arguments, estimate, screening=False):
        self.check_stop()
        self.assert_source()
        reserve = self.screen_reserve if screening else 0
        available = self.remaining() - reserve - 30  # Leave supervisor cleanup time.
        if available < estimate:
            raise BudgetStop(f"Insufficient remaining budget for {label}; confirmation reserve preserved")
        timeout = min(21600, available)
        log = self.root / "logs" / (label + ".log")
        if log.exists():
            raise RuntimeError(f"Refusing to reuse job output: {log}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.args.video_deps.resolve())
        command = [sys.executable, str(self.source / "scripts/windows_run.py"),
                   "--timeout", str(timeout), "--log", str(log), "--", *map(str, arguments)]
        self.status("running", current_job=label)
        self.ledger(event="job_started", label=label, arguments=list(map(str, arguments)), timeout=timeout)
        start = time.monotonic()
        process = subprocess.Popen(command, cwd=self.source, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        # windows_run owns descendant tracking and the hard job timeout. Heartbeat
        # here makes a detached queue's activity visible without touching Minecraft.
        while process.poll() is None:
            self.status("running")
            time.sleep(5)
        elapsed = time.monotonic() - start
        supervised = clean_supervisor(log.with_suffix(".status.json"))
        if process.returncode:
            raise RuntimeError(f"Supervisor failed for {label}: {process.returncode}")
        self.state["jobs_completed"] += 1
        self.ledger(event="job_completed", label=label, wall_seconds=elapsed, supervisor=supervised)
        return elapsed

    def train(self, name, seed, config, target, parent=None, screening=False):
        directory = self.root / "models" / f"{name}_seed{seed}"
        if directory.exists():
            raise RuntimeError(f"Training directory already exists: {directory}")
        frozen = self.root / "configs" / f"{name}_seed{seed}.json"
        expected = copy.deepcopy(config)
        expected["training"]["total_timesteps"] = target
        write_json(frozen, expected)
        args = ["-m", "combat_rl.train", "--run-dir", directory, "--steps", target, "--device", "cpu"]
        previous_steps = 0
        parent_hash = None
        if parent:
            parent = Path(parent)
            previous_steps = read_json(parent.with_suffix(".json"))["timesteps"]
            parent_hash = sha256(parent)
            args += ["--resume", parent]
        else:
            args += ["--config", frozen, "--condition", self.condition, "--seed", seed]
        estimate = max(300, (target - previous_steps) * self.rate * 1.5 + 180)
        elapsed = self.run_job(f"{name}_seed{seed}-train", args, estimate, screening)
        model = directory / "final.zip"
        identity = inspect_model(model, expected, self.condition, seed, target, self.revision, fresh=parent is None)
        if parent and (sha256(parent) != parent_hash
                       or Path(read_json(model.with_suffix(".json"))["resume_from"]).resolve() != parent.resolve()):
            raise RuntimeError("Resume provenance changed")
        self.ledger(event="checkpoint", name=name, seed=seed, parent=str(parent) if parent else None,
                    parent_sha256=parent_hash, config=expected, **identity)
        return str(model), elapsed

    def evaluate(self, name, seed, model, repeat=False, trace=False, screening=False):
        suffix = "repeat" if repeat else "validation"
        output = self.root / "evaluations" / f"{name}_seed{seed}-{suffix}.csv"
        arguments = ["-m", "combat_rl.evaluate", "--model", model, "--split", "validation",
                     "--episodes", 50, "--output", output]
        if trace:
            arguments += ["--trace-output", output.with_suffix(".trace.csv")]
        self.run_job(f"{name}_seed{seed}-{suffix}", arguments, 600, screening)
        config = read_json(Path(model).with_suffix(".json"))["config"]
        score = evaluation_score(output, model, config, self.condition, seed)
        self.ledger(event="evaluation", name=name, seed=seed, repeat=repeat, score=score, csv=str(output))
        return score

    def record(self, name, model, scenario, screening=False):
        output = self.root / "videos" / f"{name}-validation{scenario}"
        self.run_job(f"{name}-video{scenario}", ["-m", "combat_rl.record_episode", "--model", model,
                    "--output-dir", output, "--scenario-seed", scenario], 300, screening)
        sys.path.insert(0, str(self.args.video_deps.resolve()))
        import imageio_ffmpeg
        recorded = read_json(output / "episode.json")
        if recorded["model_sha256"] != sha256(model) or recorded["scenario_seed"] != scenario or (output / "INCOMPLETE.txt").exists():
            raise RuntimeError("Invalid recording identity/completion")
        for filename, count_key in (("episode.mp4", "raw_frames"), ("episode-annotated.mp4", "annotated_frames")):
            count, seconds = imageio_ffmpeg.count_frames_and_secs(str(output / filename))
            if count != recorded[count_key] or abs(seconds - count / 20) > 0.06:
                raise RuntimeError("Encoded recording frame count/duration mismatch")
        self.ledger(event="video_verified", path=str(output), metrics=recorded["metrics"])

    def screening_score(self, name, model):
        scores = [self.evaluate(name, 0, model, screening=True),
                  self.evaluate(name, 0, model, repeat=True, screening=True)]
        return average(scores)

    def save_selection(self, status, models, scores, recipe=None, **extra):
        self.selection = dict(status=status, condition=self.condition, recipe=recipe,
                              models=models, scores_by_seed=scores, aggregate=average(scores),
                              test_split_used=False, **extra)
        write_json(self.root / "selection.json", self.selection)

    def optimize(self, manifest, results, durations):
        self.config = manifest["config"]
        self.rate = max(durations) / 1_000_000
        aggregates = {c: average([r["score"] for r in results[c]]) for c in CONDITIONS}
        self.condition = max(CONDITIONS, key=lambda c: rank(aggregates[c]))
        originals = results[self.condition]
        baseline_models = [r["model"] for r in originals]
        baseline_scores = [r["score"] for r in originals]
        write_json(self.root / "comparison-summary.json", dict(conditions=aggregates, runs=results, selected=self.condition))
        self.save_selection("baseline_retained", baseline_models, baseline_scores)

        trace_score = self.evaluate("baseline", 0, baseline_models[0], trace=True, screening=True)
        # Repeat every seed-0 screening evaluation, rather than deciding after
        # observing whether a candidate happens to land near the promotion gate.
        baseline_scores[0] = average([baseline_scores[0], trace_score])
        progress = list(Path(baseline_models[0]).parent.glob("logs-*/progress.csv"))
        if len(progress) != 1:
            raise RuntimeError("Expected one fresh baseline training session")
        diagnostics = diagnostic_summary(self.root / "evaluations/baseline_seed0-validation.trace.csv", progress[0], self.config)
        hypotheses = proposed_settings(diagnostics)
        write_json(self.root / "diagnostics.json", dict(diagnostics=diagnostics, candidates=hypotheses,
                   note="Thresholds are bounded search heuristics; reward/curriculum changes require manual trace review."))
        first_seed = self.config["evaluation"]["validation_seed"]
        for scenario in range(first_seed, first_seed + 5):
            # Recording counts against the 18-hour cap; keep the confirmation reserve.
            if self.remaining() < self.screen_reserve + 600:
                raise BudgetStop("Insufficient budget for baseline recording")
            self.record("baseline", baseline_models[0], scenario, screening=True)

        candidates = []
        parent = baseline_models[0]
        continuation_seconds = 0
        try:
            for target in (1_500_000, 2_000_000):
                name = f"more_{target}"
                model, elapsed = self.train(name, 0, self.config, target, parent, screening=True)
                continuation_seconds += elapsed
                score = self.screening_score(name, model)
                candidate = dict(name=name, kind="continuation", target=target, config=self.config,
                                 model=model, score=score, train_seconds=continuation_seconds,
                                 milestones=[t for t in (1_500_000, 2_000_000) if t <= target],
                                 qualifies=qualifies(score, baseline_scores[0]))
                candidates.append(candidate)
                self.ledger(event="screened_candidate", **candidate)
                parent = model
                self.screen_reserve = max(self.screen_reserve, 2 * continuation_seconds * 1.5 + 1800)
                if (score["success_rate"] <= baseline_scores[0]["success_rate"] - 0.06
                        or score["waves"] <= baseline_scores[0]["waves"] - 0.5):
                    self.ledger(event="continuation_stopped", reason="Clear validation regression")
                    break
            for hypothesis in hypotheses:
                config = copy.deepcopy(self.config)
                config["training"].update(hypothesis["changes"])
                name = hypothesis["name"]
                self.ledger(event="hypothesis", **hypothesis)
                model, elapsed = self.train(name, 0, config, 1_000_000, screening=True)
                score = self.screening_score(name, model)
                candidate = dict(name=name, kind="fresh_settings", target=1_000_000, config=config,
                                 model=model, score=score, train_seconds=elapsed, milestones=[1_000_000],
                                 qualifies=qualifies(score, baseline_scores[0]))
                candidates.append(candidate)
                self.ledger(event="screened_candidate", **candidate)
                self.screen_reserve = max(self.screen_reserve, 2 * elapsed * 1.5 + 1800)
        except BudgetStop as error:
            self.ledger(event="screening_stopped", reason=str(error))
            # An explicit stop marker also prevents confirmation below.
        write_json(self.root / "screening.json", dict(baseline_seed0=baseline_scores[0], candidates=candidates))
        eligible = [c for c in candidates if c["qualifies"]]
        if not eligible:
            self.save_selection("baseline_retained_no_candidate_passed", baseline_models, baseline_scores)
            return
        winner = max(eligible, key=lambda c: (*rank(c["score"]), -c["target"]))
        write_json(self.root / "confirmation-recipe.json", winner)
        self.save_selection("baseline_retained_pending_confirmation", baseline_models, baseline_scores, proposed_recipe=winner)
        models, scores = [winner["model"]], [winner["score"]]
        for seed in (1, 2):
            if winner["kind"] == "continuation":
                model = baseline_models[seed]
                for milestone in winner["milestones"]:
                    model, _ = self.train(f"more_{milestone}", seed, self.config, milestone, model)
            else:
                model, _ = self.train(winner["name"], seed, winner["config"], winner["target"])
            score = self.evaluate(winner["name"], seed, model)
            models.append(model)
            scores.append(score)
            write_json(self.root / "confirmation-progress.json", dict(recipe=winner, models=models, scores=scores))
        promoted = confirmed(scores, baseline_scores)
        write_json(self.root / "confirmation.json", dict(promoted=promoted, candidate=scores,
                   baseline=baseline_scores, recipe=winner, candidate_models=models))
        if promoted:
            self.save_selection("validated_improvement", models, scores, recipe=winner,
                                baseline_scores=baseline_scores, baseline_models=baseline_models)
        else:
            self.save_selection("baseline_retained_confirmation_failed", baseline_models, baseline_scores,
                                rejected_recipe=winner, rejected_scores=scores)
        # This file locks selection before any future held-out evaluation. This
        # queue never calls the evaluator's test split and never tunes after test.
        write_json(self.root / "final-selection.json", self.selection)
        for scenario in range(first_seed, first_seed + 3):
            self.record("selected", self.selection["models"][0], scenario)

    def main(self):
        self.root.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents two background launchers from sharing outputs.
        with (self.root / "queue.lock").open("x", encoding="utf-8") as lock:
            lock.write(json.dumps({"pid": os.getpid(), "created": psutil.Process().create_time()}))
        try:
            self.assert_source()
            write_json(self.root / "manifest.json", dict(self.state, comparison_manifest_sha256=self.args.comparison_manifest_sha256,
                       predecessor_pid=self.args.comparison_pid, predecessor_created=self.args.comparison_created,
                       validation_only=True, initial_confirmation_reserve_hours=6,
                       selection_rule="completion rate, waves, kills; seed 0 screening; seeds 1/2 confirmation"))
            self.wait_for_comparison()
            manifest, results, durations = comparison_results(self.live, self.args.comparison_manifest_sha256)
            # Check the queue predecessor again after reading its artifacts.
            if self.predecessor_alive():
                raise RuntimeError("Comparison process is still running")
            self.active_start = time.monotonic()
            self.status("optimizing", started_utc=datetime.now(timezone.utc).isoformat())
            self.optimize(manifest, results, durations)
            if self.selection and not (self.root / "final-selection.json").exists():
                write_json(self.root / "final-selection.json", self.selection)
            self.status("complete", current_job=None, selection=self.selection["status"])
            return 0
        except BudgetStop as error:
            if self.selection:
                write_json(self.root / "final-selection.json", self.selection)
            self.status("stopped", reason=str(error), current_job=None)
            return 0
        except Exception as error:
            self.status("failed", error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--comparison-pid", type=int, required=True)
    parser.add_argument("--comparison-created", type=float, required=True)
    parser.add_argument("--comparison-manifest-sha256", required=True)
    parser.add_argument("--video-deps", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=18)
    parser.add_argument("--wait-hours", type=float, default=24)
    parser.add_argument("--cutoff-utc", type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    if (os.name != "nt" or not 0 < args.hours <= 18 or not 0 < args.wait_hours <= 36
            or args.cutoff_utc.tzinfo is None or args.comparison_pid <= 0):
        parser.error("Requires Windows, 0<hours<=18, 0<wait-hours<=36, and a timezone-aware cutoff")
    if args.output_root.resolve() == args.comparison_root.resolve():
        parser.error("Use a new optimization root separate from the comparison")
    return Queue(args).main()


if __name__ == "__main__":
    sys.exit(main())
