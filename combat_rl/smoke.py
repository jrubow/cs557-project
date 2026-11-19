"""Bounded integration check on real Minecraft; run before spending a training budget."""

import argparse
import json
import math
import signal
import sys
import time

import numpy as np

from .config import load_config
from .env import CombatEnv
from .policies import choose_action
from .task import NO_OP_ACTION


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_checks(config, seed, port, max_decisions):
    # Two waves are enough to exercise a genuine 1 -> 2 enemy transition.
    config["environment"]["max_waves"] = 2
    config["environment"]["max_episode_steps"] = max_decisions
    env = CombatEnv(config, condition="B", evaluation=True, port=port)
    checks = {}
    decisions = 0

    def step(action):
        nonlocal decisions
        require(decisions < max_decisions, "Smoke decision budget exhausted")
        decisions += 1
        result = env.step(action)
        require(env.observation_space.contains(result[0]), "Invalid observation values/shape")
        require(not result[2] and not result[3],
                f"Episode ended during smoke check: {result[4]['terminal_reason']}")
        return result

    def stage(name):
        print(json.dumps({"stage": name, "decisions": decisions}), file=sys.stderr, flush=True)

    def reset_check(scenario_seed):
        observation, info = env.reset(seed=scenario_seed)
        require(env.observation_space.contains(observation), "Invalid reset observation")
        require(env.state.health >= 19.99 and not env.state.is_dead, "Reset did not restore health")
        require(len(env.state.entities) == 1, "Reset did not create exactly one enemy")
        require(info["kills"] == 0 and info["damage_dealt"] == info["damage_taken"] == 0,
                "Episode metrics were not reset")
        # Compare against the seeded spawn request, allowing a little live mob movement.
        rng = np.random.default_rng(scenario_seed)
        yaw = rng.uniform(-180, 180)
        max_radius = config["environment"]["arena_radius"] - 2
        radius = rng.uniform(min(4, max_radius), max_radius)
        angle = rng.uniform(-math.pi, math.pi)
        x = config["environment"]["arena_center_x"] + radius * math.cos(angle)
        z = config["environment"]["arena_center_z"] + radius * math.sin(angle)
        enemy = env.state.entities[0]
        error = math.hypot(enemy.x - x, enemy.z - z)
        require(error < 1.25, f"Spawn disagrees with requested seed by {error:.2f} blocks")
        require(abs((env.state.yaw - yaw + 180) % 360 - 180) < 1,
                "Initial orientation disagrees with requested seed")
        return observation, error

    try:
        stage("initial_and_fast_reset")
        _, first_error = reset_check(seed)
        observation, second_error = reset_check(seed)
        checks["seeded_resets"] = {"spawn_errors_blocks": [first_error, second_error]}

        stage("movement_and_camera")
        # Freeze the enemy only while checking controls and reset-counter isolation.
        env._raw.add_commands([
            "execute as @e[type=minecraft:zombie] run data merge entity @s {NoAI:1b}"
        ])
        for _ in range(4):
            observation, _, _, _, info = step(NO_OP_ACTION)
        require(info["damage_taken"] == 0 and info["damage_dealt"] == 0 and info["kills"] == 0,
                "Reset/setup statistics leaked into episode rewards")
        x, z, yaw, pitch = env.state.x, env.state.z, env.state.yaw, env.state.pitch
        action = NO_OP_ACTION.copy()
        action[0] = 1
        for _ in range(3):
            observation, *_ = step(action)
        displacement = math.hypot(env.state.x - x, env.state.z - z)
        require(displacement > 0.05, "Forward action did not move the player")
        action = NO_OP_ACTION.copy()
        action[2], action[3] = 4, 2  # +30 yaw, +10 pitch, once per decision.
        observation, *_ = step(action)
        yaw_change = (env.state.yaw - yaw + 180) % 360 - 180
        pitch_change = env.state.pitch - pitch
        require(5 < yaw_change < 60 and 2 < pitch_change < 25,
                f"Unexpected camera response: yaw={yaw_change}, pitch={pitch_change}")
        checks["controls"] = {"movement_blocks": displacement, "yaw_degrees": yaw_change,
                              "pitch_degrees": pitch_change}

        stage("actual_enemy_damage")
        observation, _ = reset_check(seed + 1)
        x, z = config["environment"]["arena_center_x"], config["environment"]["arena_center_z"]
        env._raw.add_commands([
            f"tp @p {float(x)} 65.0 {float(z)} 0 0",
            f"tp @e[type=minecraft:zombie,limit=1] {float(x)} 65.0 {z + 1.5}",
        ])
        initial_health = env.state.health
        for _ in range(min(100, max_decisions)):
            observation, _, _, _, info = step(NO_OP_ACTION)
            if env.state.health < initial_health and info["damage_taken"] > 0:
                break
        else:
            raise RuntimeError("No real zombie hit with matching damage_taken statistic")
        damaged_health = env.state.health
        checks["received_damage"] = {"health_lost": initial_health - damaged_health,
                                     "statistic_health_points": info["damage_taken"]}

        stage("actual_attack_kill_and_wave_transition")
        # Position/freeze a full-health target to avoid making this a policy-quality test.
        # No /damage, /kill, health edits, or weapon buffs are used to pass combat checks.
        enemy_uuid = env.state.entities[0].uuid
        env._raw.add_commands([
            "execute as @e[type=minecraft:zombie] run data merge entity @s {NoAI:1b}",
            f"tp @p {float(x)} 65.0 {float(z)} 0 0",
            f"tp @e[type=minecraft:zombie,limit=1] {float(x)} 65.0 {z + 2.2}",
        ])
        observation, *_ = step(NO_OP_ACTION)
        previous_enemy_health = {e.uuid: e.health for e in env.state.entities}
        saw_enemy_health_drop = False
        local_step = 0
        while decisions < max_decisions:
            action = choose_action(observation, local_step, config)
            observation, _, _, _, info = step(action)
            current_health = {e.uuid: e.health for e in env.state.entities}
            if (enemy_uuid in current_health and enemy_uuid in previous_enemy_health
                    and current_health[enemy_uuid] < previous_enemy_health[enemy_uuid]):
                saw_enemy_health_drop = True
            previous_enemy_health = current_health
            local_step += 1
            if info["waves_completed"] >= 1:
                break
        require(saw_enemy_health_drop and info["damage_dealt"] > 0,
                "No actual sword hit with both enemy-health loss and damage_dealt statistic")
        require(info["kills"] >= 1 and info["waves_completed"] == 1 and env.wave == 2,
                "A real player kill did not advance wave 1 to wave 2")
        for _ in range(15):
            if len(env.state.entities) == 2:
                break
            observation, _, _, _, info = step(NO_OP_ACTION)
        require(len(env.state.entities) == 2, "Wave 2 did not spawn two living zombies")
        require(env.state.health <= damaged_health + 0.01,
                "Player healed unexpectedly between waves")
        checks["combat"] = {"damage_dealt": info["damage_dealt"], "kills": info["kills"],
                            "waves_completed": info["waves_completed"],
                            "next_wave_enemies": len(env.state.entities),
                            "health_after_transition": env.state.health}

        stage("reset_after_combat")
        reset_check(seed)
        checks["reset_after_combat"] = True
        return {"passed": True, "decisions": decisions, "checks": checks,
                "scope": "Live Minecraft controls, counters, real damage/kill and 1-to-2 wave transition.",
                "controlled_setup": "Enemies frozen/teleported for controls and aiming; health unchanged."}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=557000)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=int, default=900,
                        help="Wall-clock limit, including the first Minecraft build/startup (Unix)")
    args = parser.parse_args()
    if args.max_decisions < 30 or args.timeout_seconds < 1:
        parser.error("max-decisions must be at least 30 and timeout-seconds must be positive")
    start = time.perf_counter()

    def timeout(_signal, _frame):
        raise TimeoutError("Minecraft smoke check exceeded its wall-clock budget")

    timer_available = hasattr(signal, "SIGALRM")
    if timer_available:
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.timeout_seconds)
    try:
        result = run_checks(load_config(args.config), args.seed, args.port, args.max_decisions)
        result["wall_seconds"] = time.perf_counter() - start
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error),
                          "wall_seconds": time.perf_counter() - start}), flush=True)
        raise SystemExit(1) from error
    finally:
        if timer_available:
            signal.alarm(0)


if __name__ == "__main__":
    main()
