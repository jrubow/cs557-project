"""Train the same small PPO policy for each reward/information condition."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

from combat_rl.config import load_config


def read_checkpoint(path):
    path = Path(path).with_suffix(".zip")
    return path, json.loads(path.with_suffix(".json").read_text())


FINETUNE_FIELDS = {"learning_rate", "gamma", "ent_coef", "clip_range"}


def finetune_changes(previous, proposed):
    """Allow explicit reward/PPO adaptation while preserving the original task."""
    changes = {}
    for section in previous.keys() | proposed.keys():
        old, new = previous.get(section, {}), proposed.get(section, {})
        for key in old.keys() | new.keys():
            if old.get(key) == new.get(key):
                continue
            allowed = section == "reward" and key in old and key in new
            allowed |= section == "training" and key in FINETUNE_FIELDS | {"total_timesteps"}
            if not allowed:
                raise ValueError(f"Fine-tuning cannot change {section}.{key}")
            value = new[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid fine-tuning value: {section}.{key}")
            if section == "training":
                if (key in {"learning_rate", "total_timesteps"} and value <= 0
                        or key in {"gamma", "clip_range"} and not 0 < value <= 1
                        or key == "ent_coef" and value < 0):
                    raise ValueError(f"Invalid fine-tuning value: {section}.{key}")
            changes[f"{section}.{key}"] = {"before": old.get(key), "after": value}
    return changes


def finetune_ppo_kwargs(config):
    # SB3 applies these before _setup_model: schedules and rollout-buffer gamma
    # are rebuilt, then saved policy AND optimizer tensors are restored.
    return {key: config["training"][key] for key in FINETUNE_FIELDS}


def save_checkpoint(model, path, metadata):
    path = Path(path).with_suffix(".zip")
    temporary = path.with_name(path.stem + ".tmp.zip")
    model.save(temporary)
    temporary.replace(path)
    record = dict(metadata, timesteps=int(model.num_timesteps))
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")


def runtime_metadata():
    versions = {}
    for package in ("craftground", "stable-baselines3", "torch", "gymnasium", "numpy", "protobuf"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {"python": sys.version, "platform": platform.platform(), "packages": versions}
    root = Path(__file__).resolve().parents[1]
    for key, command in (
        ("git_revision", ["git", "rev-parse", "HEAD"]),
        ("git_status", ["git", "status", "--porcelain"]),
        ("pip_freeze", [sys.executable, "-m", "pip", "freeze"]),
    ):
        try:
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=15)
            result[key] = completed.stdout.strip() if completed.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result[key] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--condition", choices=["A", "B", "C"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, help="Target total steps, including a resumed run")
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--finetune-config", type=Path,
                        help="Explicit reward/PPO adaptation from --resume into a NEW run directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.finetune_config and (not args.resume or args.config):
        parser.error("--finetune-config requires --resume and cannot be combined with --config")
    if args.finetune_config and args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("Fine-tuning requires a new, empty run directory")

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from combat_rl.env import CombatEnv

    if args.resume:
        checkpoint, previous = read_checkpoint(args.resume)
        config = previous["config"]
        if args.config and load_config(args.config) != config:
            parser.error("Resume with the saved config; use --steps to extend the budget")
        if args.condition and args.condition != previous["condition"]:
            parser.error("Resume condition does not match checkpoint")
        if args.seed is not None and args.seed != previous["seed"]:
            parser.error("Resume seed does not match checkpoint")
        condition, seed = previous["condition"], previous["seed"]
        if args.n_envs is not None and args.n_envs != config["training"]["n_envs"]:
            parser.error("Keep the same --n-envs when resuming")
        if args.finetune_config:
            config = load_config(args.finetune_config)
            try:
                finetune_changes(previous["config"], config)
            except ValueError as error:
                parser.error(str(error))
    else:
        config = load_config(args.config)
        condition = args.condition or "B"
        seed = args.seed if args.seed is not None else 0
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            parser.error("Run directory is not empty; choose a new directory or use --resume")
    training = config["training"]
    if args.steps is not None:
        training["total_timesteps"] = args.steps
    if args.n_envs is not None:
        training["n_envs"] = args.n_envs
    if training["n_envs"] != 1:
        parser.error("Use n-envs=1: CraftGround instances also share game files, not just ports")
    if training["total_timesteps"] < 1:
        parser.error("steps must be positive")
    rollout_size = training["n_envs"] * training["n_steps"]
    if rollout_size < 2 or not 1 < training["batch_size"] <= rollout_size:
        parser.error("Require 1 < batch_size <= n_envs * n_steps")
    if rollout_size % training["batch_size"]:
        parser.error("batch_size must divide n_envs * n_steps")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    session = time.strftime("%Y%m%d-%H%M%S")
    log_dir = args.run_dir / ("logs-" + session)
    log_dir.mkdir()
    metadata = {"condition": condition, "seed": seed, "config": config,
                "algorithm": "PPO", "device": args.device,
                "resume_from": str(args.resume) if args.resume else None,
                "runtime": runtime_metadata()}
    if args.resume:
        metadata["resume_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if args.finetune_config:
        metadata["fine_tuning"] = {"changes": finetune_changes(previous["config"], config),
                                   "parent_timesteps": previous["timesteps"],
                                   "optimizer_restored": True}
    (args.run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    torch.set_num_threads(training["torch_threads"])

    def make_env(rank):
        def create():
            env = CombatEnv(config, condition=condition, port=args.port + rank)
            if args.resume:
                env.set_training_step(previous["timesteps"])
            return Monitor(env, str(log_dir / f"env{rank}.monitor.csv"),
                           info_keywords=("waves_completed", "success", "kills", "wave_target"))
        return create

    count = training["n_envs"]
    env_fns = [make_env(rank) for rank in range(count)]
    vec_env = DummyVecEnv(env_fns)
    model = None
    interrupted = False
    previous_handler = signal.getsignal(signal.SIGTERM)

    def stop(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)

    class TrainingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last_saved = 0

        def _on_training_start(self):
            self.last_saved = self.num_timesteps

        def _on_step(self):
            self.training_env.env_method("set_training_step", self.num_timesteps)
            return True

        def _on_rollout_start(self):
            # At this point the previous rollout has already updated the policy.
            episodes = self.model.ep_info_buffer
            if episodes:
                for metric in ("waves_completed", "success", "kills", "wave_target"):
                    self.logger.record("combat/" + metric,
                                       sum(float(row[metric]) for row in episodes) / len(episodes))
            if self.num_timesteps - self.last_saved >= training["checkpoint_every"]:
                save_checkpoint(self.model, args.run_dir / f"step_{self.num_timesteps}", metadata)
                self.last_saved = self.num_timesteps

    try:
        vec_env.seed(seed)
        if args.resume:
            overrides = finetune_ppo_kwargs(config) if args.finetune_config else {}
            model = PPO.load(checkpoint, env=vec_env, device=args.device, **overrides)
            vec_env.env_method("set_training_step", model.num_timesteps)
        else:
            model = PPO("MlpPolicy", vec_env, seed=seed, device=args.device, verbose=1,
                        n_steps=training["n_steps"], batch_size=training["batch_size"],
                        learning_rate=training["learning_rate"], gamma=training["gamma"],
                        gae_lambda=training["gae_lambda"], ent_coef=training["ent_coef"],
                        clip_range=training["clip_range"], n_epochs=training["n_epochs"],
                        policy_kwargs={"net_arch": training["policy_layers"]})
            initial_bias = training.get("initial_action_bias", 0.0)
            if initial_bias:
                from combat_rl.task import COMPACT_NVECS, NO_OP_ACTION
                preferred = NO_OP_ACTION.copy()
                preferred[-1] = 1  # Attack while initially favoring steady aim/movement.
                with torch.no_grad():
                    offset = 0
                    for choices, selected in zip(COMPACT_NVECS, preferred):
                        model.policy.action_net.bias[offset + int(selected)] += initial_bias
                        offset += int(choices)
                print(f"Initial action preference logit bias: {initial_bias}")
        model.set_logger(configure(str(log_dir), ["stdout", "csv"]))
        remaining = training["total_timesteps"] - model.num_timesteps
        if remaining > 0:
            model.learn(total_timesteps=remaining, reset_num_timesteps=False,
                        callback=TrainingCallback())
        save_checkpoint(model, args.run_dir / "final", metadata)
    except KeyboardInterrupt:
        interrupted = True
        if model is not None:
            try:
                save_checkpoint(model, args.run_dir / "interrupted", metadata)
                print("Saved interrupted.zip. Resume with --resume and the same --run-dir.")
            except Exception as error:
                print(f"Could not save interrupted checkpoint: {error}", file=sys.stderr)
    except Exception:
        if model is not None:
            try:
                save_checkpoint(model, args.run_dir / "failed", metadata)
                print("Saved failed.zip before reporting the training error.", file=sys.stderr)
            except Exception as error:
                print(f"Could not save failure checkpoint: {error}", file=sys.stderr)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        try:
            vec_env.close()
        except Exception as error:
            print(f"Could not close simulator cleanly: {error}", file=sys.stderr)
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
