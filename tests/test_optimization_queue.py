"""Offline orchestration checks. No Minecraft processes or PPO training are started."""
import argparse
import copy
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from scripts import queue_optimization_windows as q


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/comparison.json").read_text())


def checkpoint(path, condition, seed, config, target, revision, parent=None):
    config = copy.deepcopy(config)
    config["training"]["total_timesteps"] = target
    steps = ((target + 1023) // 1024) * 1024
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": steps}))
    q.write_json(path.with_suffix(".json"), dict(condition=condition, seed=seed, config=config,
                 timesteps=steps, resume_from=str(parent) if parent else None,
                 runtime=dict(git_revision=revision, git_status="")))


def evaluation(path, model, waves, trace=False):
    meta = q.read_json(model.with_suffix(".json"))
    rows = []
    for index in range(50):
        completed = waves[index % len(waves)]
        rows.append(dict(policy="ppo", condition=meta["condition"], training_seed=meta["seed"],
            model=str(model), split="validation", scenario_seed=50000 + index, wave_target=5,
            success=completed == 5, waves_completed=completed, kills=completed * (completed + 1) // 2,
            damage_dealt=completed * 20, damage_taken=20, elapsed_steps=1, elapsed_seconds=0.2,
            episode_return=0, terminal_reason="success" if completed == 5 else "death"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    q.write_json(path.with_suffix(".json"), dict(policy="ppo", condition=meta["condition"], training_seed=meta["seed"],
        model=str(model), split="validation", first_seed=50000, episodes=50, deterministic=True, config=meta["config"]))
    if trace:
        traces = [dict(scenario_seed=r["scenario_seed"], decision=0, wave=1, health=20, x=0, z=0,
                  nearest_distance=2, nearest_bearing=0, move=0, strafe=0, attack=1,
                  damage_dealt_delta=r["damage_dealt"], damage_taken_delta=20,
                  kills_delta=r["kills"], reward=0) for r in rows]
        with path.with_suffix(".trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=traces[0])
            writer.writeheader()
            writer.writerows(traces)


def comparison(root):
    q.write_json(root / "manifest.json", dict(source_revision="baseline", config=CONFIG,
                 steps_per_run=1000000, validation_episodes=50))
    for condition in q.CONDITIONS:
        for seed in q.SEEDS:
            name = f"{condition}_seed{seed}"
            model = root / name / "final.zip"
            checkpoint(model, condition, seed, CONFIG, 1000000, "baseline")
            evaluation(root / name / "validation.csv", model, [2, 3] if condition == "B" else [2])
            for stage in ("train", "validation"):
                q.write_json(root / "logs" / f"{name}-{stage}-fake.status.json",
                             dict(exit_code=0, remaining_descendant_pids=[], wall_seconds=6000))
            progress = root / name / "logs-fake/progress.csv"
            progress.parent.mkdir()
            progress.write_text("train/approx_kl,train/clip_fraction,train/entropy_loss,train/explained_variance\n"
                                "0.04,0.35,-5.0,0.9\n")
    return q.sha256(root / "manifest.json")


class FakeQueue(q.Queue):
    def git(self, *args):
        return "optimization" if args[0] == "rev-parse" else ""

    def run_job(self, label, arguments, estimate, screening=False):
        self.check_stop()
        arguments = list(map(str, arguments))
        self.calls.append((label, arguments))
        def arg(name):
            return arguments[arguments.index(name) + 1]
        if arguments[1] == "combat_rl.train":
            parent = Path(arg("--resume")) if "--resume" in arguments else None
            if parent:
                previous = q.read_json(parent.with_suffix(".json"))
                config, seed = previous["config"], previous["seed"]
            else:
                config, seed = q.read_json(arg("--config")), int(arg("--seed"))
            checkpoint(Path(arg("--run-dir")) / "final.zip", self.condition, seed, config,
                       int(arg("--steps")), "optimization", parent)
        elif arguments[1] == "combat_rl.evaluate":
            model = Path(arg("--model"))
            seed = q.read_json(model.with_suffix(".json"))["seed"]
            if label.startswith("baseline"):
                waves = [2, 3]
            elif self.reject_confirmation and seed:
                waves = [1, 2]
            elif label.startswith("lower_lr"):
                waves = [3, 4] if not self.continuation_wins else [3]
            elif label.startswith("more_2000000"):
                waves = [4] if self.continuation_wins else [3]
            elif label.startswith("more_"):
                waves = [3]
            else:
                waves = [2, 3]
            evaluation(Path(arg("--output")), model, waves, "--trace-output" in arguments)
        else:
            raise AssertionError(arguments)
        return 4000

    def record(self, name, model, scenario, screening=False):
        self.check_stop()
        self.recordings.append((name, str(model), scenario))


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.live = self.root / "comparison"
        self.digest = comparison(self.live)
        self.args = argparse.Namespace(output_root=self.root / "optimization", comparison_root=self.live,
            video_deps=self.root, hours=18, wait_hours=24, comparison_pid=1, comparison_created=1,
            comparison_manifest_sha256=self.digest, cutoff_utc=datetime.now(timezone.utc) + timedelta(days=1))
        self.queue = FakeQueue(self.args)
        self.queue.root.mkdir()
        self.queue.active_start = time.monotonic()
        self.queue.calls, self.queue.recordings = [], []
        self.queue.reject_confirmation = self.queue.continuation_wins = False

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_comparison_rejects_missing_or_unclean_job(self):
        manifest, results, durations = q.comparison_results(self.live, self.digest)
        self.assertEqual(len(results["B"]), 3)
        self.assertEqual(len(durations), 9)
        broken = self.live / "logs/C_seed2-validation-fake.status.json"
        q.write_json(broken, dict(exit_code=0, remaining_descendant_pids=[123]))
        with self.assertRaisesRegex(RuntimeError, "unclean"):
            q.comparison_results(self.live, self.digest)
        broken.unlink()
        with self.assertRaisesRegex(RuntimeError, "one successful"):
            q.comparison_results(self.live, self.digest)

    def test_checkpoint_and_validation_identity_are_checked(self):
        model = self.live / "B_seed1/final.zip"
        q.inspect_model(model, CONFIG, "B", 1, 1000000, "baseline", fresh=True)
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            q.inspect_model(model, CONFIG, "B", 2, 1000000, "baseline")
        path = self.live / "B_seed1/validation.csv"
        meta = q.read_json(path.with_suffix(".json"))
        meta["split"] = "test"
        q.write_json(path.with_suffix(".json"), meta)
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            q.evaluation_score(path, model, CONFIG, "B", 1)

    def test_handoff_rejects_dead_predecessor_without_completion(self):
        q.write_json(self.live / "status.json", dict(phase="training", completed=[]))
        with patch.object(self.queue, "predecessor_alive", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "without.*successful"):
                self.queue.wait_for_comparison()
        q.write_json(self.live / "status.json", dict(phase="complete",
                     completed=[f"{c}_seed{s}" for s in (0, 2) for c in q.CONDITIONS]))
        with patch.object(self.queue, "predecessor_alive", return_value=False):
            self.queue.wait_for_comparison()
        self.assertFalse(self.queue.calls)

    def test_budget_and_stop_marker_prevent_process_launch(self):
        self.queue.active_start -= 13 * 3600
        with patch.object(q.subprocess, "Popen") as popen:
            with self.assertRaises(q.BudgetStop):
                q.Queue.run_job(self.queue, "forbidden", ["-m", "combat_rl.train"], 600, screening=True)
            popen.assert_not_called()
        (self.queue.root / "STOP_AFTER_CURRENT").touch()
        with self.assertRaises(q.BudgetStop):
            self.queue.check_stop()

    def test_settings_winner_is_confirmed_without_test_split(self):
        self.queue.optimize(*q.comparison_results(self.live, self.digest))
        self.assertEqual(self.queue.selection["status"], "validated_improvement")
        self.assertEqual(self.queue.selection["recipe"]["name"], "lower_lr")
        self.assertEqual(len(self.queue.selection["models"]), 3)
        for label, arguments in self.queue.calls:
            if "--split" in arguments:
                self.assertEqual(arguments[arguments.index("--split") + 1], "validation")
        self.assertEqual(len(self.queue.recordings), 8)

    def test_confirmation_failure_preserves_baseline_models(self):
        self.queue.reject_confirmation = True
        self.queue.optimize(*q.comparison_results(self.live, self.digest))
        self.assertEqual(self.queue.selection["status"], "baseline_retained_confirmation_failed")
        self.assertTrue(all(Path(model).parent.parent.resolve() == self.live.resolve()
                            for model in self.queue.selection["models"]))

    def test_continuation_reproduces_both_resume_milestones(self):
        self.queue.continuation_wins = True
        self.queue.optimize(*q.comparison_results(self.live, self.digest))
        self.assertEqual(self.queue.selection["recipe"]["target"], 2000000)
        for seed in (1, 2):
            label = f"more_2000000_seed{seed}-train"
            command = next(args for name, args in self.queue.calls if name == label)
            self.assertIn(f"more_1500000_seed{seed}", command[command.index("--resume") + 1])

    def test_one_lucky_training_seed_does_not_confirm(self):
        def score(waves):
            return dict(success_rate=0, waves=waves, kills=waves, damage_dealt=0, damage_taken=0)
        self.assertFalse(q.confirmed([score(4), score(2), score(2)], [score(2)] * 3))
        self.assertTrue(q.confirmed([score(3), score(3), score(2)], [score(2)] * 3))


if __name__ == "__main__":
    unittest.main()
