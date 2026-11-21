"""Evaluate PPO/random controllers on shared seeds, or measure simulator speed."""

import argparse
from contextlib import ExitStack
import csv
import json
import math
import time
from pathlib import Path

from combat_rl.config import load_config
from combat_rl.train import read_checkpoint


METRICS = ["waves_completed", "success", "kills", "damage_dealt", "damage_taken",
           "elapsed_steps", "elapsed_seconds", "terminal_reason", "wave_target"]


def scenario_start(config, split, episodes, override=None):
    """Keep development ranges below the held-out boundary, even with overrides."""
    first = config["evaluation"][split + "_seed"] if override is None else override
    lower = config["evaluation"][split + "_seed"]
    if episodes < 1 or first < lower:
        raise ValueError("Scenario range starts before the configured split or is empty")
    if split == "validation" and first + episodes > config["evaluation"]["test_seed"]:
        raise ValueError("Validation scenarios must stay below the test boundary")
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--policy", choices=["ppo", "random", "heuristic"], default=None)
    parser.add_argument("--condition", choices=["A", "B", "C"])
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--first-seed", type=int, help="Explicit fresh scenario range within the chosen split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path,
                        help="Optional per-decision CSV for post-hoc behavior inspection")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=0,
                        help="Measure this many random actions instead of evaluating episodes")
    args = parser.parse_args()
    policy = args.policy or ("ppo" if args.model else "random")
    if policy == "ppo" and not args.model:
        parser.error("--policy ppo requires --model")
    if policy != "ppo" and args.model:
        parser.error("--model can only be combined with --policy ppo")
    if args.output.exists():
        parser.error("Output exists; choose a new path to keep earlier results")
    if args.trace_output and (args.trace_output.exists() or args.trace_output.resolve() in
                             {args.output.resolve(), args.output.with_suffix(".json").resolve()}):
        parser.error("Trace must be a new path distinct from episode CSV and metadata JSON")
    if args.benchmark_steps < 0 or (args.episodes is not None and args.episodes < 1):
        parser.error("benchmark-steps must be nonnegative and episodes positive")
    if args.benchmark_steps and args.model:
        parser.error("Benchmark uses a random policy; omit --model")
    if args.benchmark_steps and args.trace_output:
        parser.error("Trace recording is for episode evaluation, not the throughput benchmark")

    from combat_rl.env import CombatEnv
    from combat_rl.policies import choose_action

    model = None
    training_seed = ""
    if args.model:
        checkpoint, metadata = read_checkpoint(args.model)
        config, condition = metadata["config"], metadata["condition"]
        training_seed = metadata["seed"]
        if args.config and load_config(args.config) != config:
            parser.error("Evaluation config must match the model's saved config")
        if args.condition and args.condition != condition:
            parser.error("Condition must match the model's saved condition")
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(config["training"]["torch_threads"])
        model = PPO.load(checkpoint, device=args.device)
    else:
        config = load_config(args.config)
        condition = args.condition or "B"
    count = args.episodes or config["evaluation"]["episodes"]
    try:
        first_seed = scenario_start(config, args.split, count, args.first_seed)
    except ValueError as error:
        parser.error(str(error))
    if model is not None:
        model.set_random_seed(first_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    env = CombatEnv(config, condition=condition, evaluation=True, port=args.port)
    try:
        if args.benchmark_steps:
            env.action_space.seed(first_seed)
            _, _ = env.reset(seed=first_seed)
            startup = time.perf_counter() - start
            start = time.perf_counter()
            episodes = 0
            for _ in range(args.benchmark_steps):
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    episodes += 1
                    env.reset(seed=first_seed + episodes)
            elapsed = time.perf_counter() - start
            rate = args.benchmark_steps / elapsed
            result = {"steps": args.benchmark_steps, "wall_seconds": elapsed,
                      "startup_seconds": startup, "steps_per_second": rate,
                      "hours_per_million_steps": 1_000_000 / rate / 3600,
                      "completed_episodes": episodes, "config": config,
                      "note": "One simulator; random actions; includes episode resets, excludes PPO updates."}
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            return

        fields = ["policy", "condition", "training_seed", "model", "split", "scenario_seed",
                  "episode_return", *METRICS]
        with ExitStack() as stack:
            stream = stack.enter_context(args.output.open("w", newline=""))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            trace_writer = None
            if args.trace_output:
                args.trace_output.parent.mkdir(parents=True, exist_ok=True)
                trace_stream = stack.enter_context(args.trace_output.open("x", newline=""))
                trace_fields = ["scenario_seed", "decision", "wave", "health", "x", "z", "yaw", "pitch",
                                "nearest_distance", "nearest_bearing", "nearest_pitch_error",
                                "move", "strafe", "yaw_action", "pitch_action", "jump", "sprint", "attack",
                                "reward", "damage_dealt_delta", "damage_taken_delta", "kills_delta"]
                trace_writer = csv.DictWriter(trace_stream, fieldnames=trace_fields)
                trace_writer.writeheader()
            for episode in range(count):
                scenario_seed = first_seed + episode
                env.action_space.seed(scenario_seed)
                observation, _ = env.reset(seed=scenario_seed)
                episode_return = 0.0
                step = 0
                while True:
                    if policy == "heuristic":
                        action = choose_action(observation, step, config)
                    elif model is None:
                        action = env.action_space.sample()
                    else:
                        action, _ = model.predict(observation, deterministic=not args.stochastic)
                    if trace_writer is not None:
                        before = env.state
                        trace = dict(scenario_seed=scenario_seed, decision=step, wave=env.wave,
                                     health=before.health, x=before.x, z=before.z,
                                     yaw=before.yaw, pitch=before.pitch,
                                     nearest_distance="", nearest_bearing="", nearest_pitch_error="")
                        # Ground truth is diagnostic only, never added to policy observations.
                        if before.entities:
                            nearest = min(before.entities, key=lambda e: (e.x-before.x)**2 + (e.z-before.z)**2)
                            dx, dz = nearest.x-before.x, nearest.z-before.z
                            yaw = math.radians(before.yaw)
                            distance = math.hypot(dx, dz)
                            trace.update(nearest_distance=distance,
                                         nearest_bearing=math.degrees(math.atan2(
                                             -math.cos(yaw)*dx - math.sin(yaw)*dz,
                                             -math.sin(yaw)*dx + math.cos(yaw)*dz)),
                                         nearest_pitch_error=math.degrees(math.atan2(
                                             before.y+1.62-(nearest.y+1.0), max(distance, 0.01))) - before.pitch)
                        trace.update(zip(["move", "strafe", "yaw_action", "pitch_action", "jump", "sprint", "attack"],
                                         map(int, action)))
                    observation, reward, terminated, truncated, info = env.step(action)
                    if trace_writer is not None:
                        trace.update(reward=float(reward),
                                     damage_dealt_delta=max(0.0, env.state.damage_dealt-before.damage_dealt),
                                     damage_taken_delta=max(0.0, env.state.damage_taken-before.damage_taken),
                                     kills_delta=max(0, env.state.kills-before.kills))
                        trace_writer.writerow(trace)
                    episode_return += float(reward)
                    step += 1
                    if terminated or truncated:
                        break
                row = {"policy": policy, "condition": condition,
                       "training_seed": training_seed, "model": str(args.model or ""),
                       "split": args.split, "scenario_seed": scenario_seed,
                       "episode_return": episode_return}
                row.update({key: info[key] for key in METRICS})
                writer.writerow(row)
                stream.flush()
                if trace_writer is not None:
                    trace_stream.flush()
                print(f"{episode + 1}/{count}: seed={scenario_seed}, "
                      f"waves={info['waves_completed']}, success={info['success']}", flush=True)
        args.output.with_suffix(".json").write_text(json.dumps({
            "policy": policy, "condition": condition, "model": str(args.model or ""),
            "training_seed": training_seed, "split": args.split, "first_seed": first_seed,
            "first_seed_override": args.first_seed,
            "episodes": count, "deterministic": not args.stochastic, "config": config,
            "trace_output": str(args.trace_output) if args.trace_output else None,
            "wall_seconds": time.perf_counter() - start}, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
