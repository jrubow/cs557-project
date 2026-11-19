"""Evaluate PPO/random controllers on shared seeds, or measure simulator speed."""

import argparse
import csv
import json
import time
from pathlib import Path

from combat_rl.config import load_config
from combat_rl.train import read_checkpoint


METRICS = ["waves_completed", "success", "kills", "damage_dealt", "damage_taken",
           "elapsed_steps", "elapsed_seconds", "terminal_reason", "wave_target"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--policy", choices=["ppo", "random", "heuristic"], default=None)
    parser.add_argument("--condition", choices=["A", "B", "C"])
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output", type=Path, required=True)
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
    if args.benchmark_steps < 0 or (args.episodes is not None and args.episodes < 1):
        parser.error("benchmark-steps must be nonnegative and episodes positive")
    if args.benchmark_steps and args.model:
        parser.error("Benchmark uses a random policy; omit --model")

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
    first_seed = config["evaluation"][args.split + "_seed"]
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
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
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
                    observation, reward, terminated, truncated, info = env.step(action)
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
                print(f"{episode + 1}/{count}: seed={scenario_seed}, "
                      f"waves={info['waves_completed']}, success={info['success']}", flush=True)
        args.output.with_suffix(".json").write_text(json.dumps({
            "policy": policy, "condition": condition, "model": str(args.model or ""),
            "training_seed": training_seed, "split": args.split, "first_seed": first_seed,
            "episodes": count, "deterministic": not args.stochastic, "config": config,
            "wall_seconds": time.perf_counter() - start}, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
