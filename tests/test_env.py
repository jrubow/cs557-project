"""Exercise task boundaries without launching Minecraft or a training job."""

from types import SimpleNamespace
import math
import unittest

import numpy as np

from combat_rl.config import load_config
from combat_rl.env import CombatEnv, ENTITY_RANGE
from combat_rl.task import NO_OP_ACTION


def observation(count=0, kills=10, dealt=100, taken=80, health=20):
    entities = [SimpleNamespace(unique_name=str(i), translation_key="entity.minecraft.zombie",
                                x=float(i + 5), y=65, z=0, health=20) for i in range(count)]
    return {"full": SimpleNamespace(x=0, y=65, z=0, yaw=0, pitch=0, health=health,
                                   is_dead=health <= 0,
                                   misc_statistics={"damage_dealt": dealt, "damage_taken": taken},
                                   killed_statistics={"zombie": kills},
                                   surrounding_entities={ENTITY_RANGE: SimpleNamespace(entities=entities)})}


class Backend:
    def __init__(self):
        self.commands, self.actions, self.script = [], [], []
        self.current = observation()

    def reset(self, **kwargs):
        self.current = observation()
        return self.current, {}

    def add_commands(self, commands):
        self.commands.extend(commands)
        count = sum(command.startswith("summon minecraft:zombie ") for command in commands)
        if count:
            self.current = observation(count)

    def step(self, action):
        self.actions.append(dict(action))
        if self.script:
            self.current = self.script.pop(0)
        return self.current, 0, False, False, {}

    def close(self):
        pass


class EnvironmentTests(unittest.TestCase):
    def test_encounter_curriculum_is_common_and_excluded_from_evaluation(self):
        config = load_config()
        config["environment"]["encounter_curriculum"] = [[0, 2.5, 3.0, 10], [200000, 4, 10, 180]]
        commands = []
        for condition in ("A", "B", "C"):
            backend = Backend()
            env = CombatEnv(config, condition=condition, backend=backend)
            env.reset(seed=42)
            commands.append(backend.commands)
        self.assertEqual(commands[0], commands[1])
        self.assertEqual(commands[1], commands[2])
        spawn = next(c for c in commands[0] if c.startswith("summon")).split()
        x, z = float(spawn[2]), float(spawn[4])
        self.assertTrue(2.499 <= math.hypot(x, z) <= 3.001)
        heading = [c for c in commands[0] if c.startswith("tp @p")][-1].split()
        yaw = float(heading[5])
        self.assertLessEqual(abs((yaw-math.degrees(math.atan2(-x, z))+180) % 360-180), 10.1)
        original = Backend()
        CombatEnv(load_config(), evaluation=True, backend=original).reset(seed=42)
        for evaluation, step in [(True, 0), (False, 200000)]:
            backend = Backend()
            env = CombatEnv(config, evaluation=evaluation, backend=backend)
            env.set_training_step(step)
            env.reset(seed=42)
            self.assertEqual(backend.commands, original.commands)

    def make_env(self):
        config = load_config()
        config["environment"].update(action_repeat=2, max_waves=2)
        backend = Backend()
        return CombatEnv(config, evaluation=True, backend=backend), backend

    def test_reset_stats_history_waves_and_attack_release(self):
        env, backend = self.make_env()
        obs, _ = env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(obs))
        first_spawn = [c for c in backend.commands if c.startswith("summon")]
        env.reset(seed=42)
        self.assertEqual(first_spawn, [c for c in backend.commands if c.startswith("summon")][-1:])
        backend.script = [observation(kills=11, dealt=300, taken=110, health=17),
                          observation(2, kills=11, dealt=300, taken=110, health=17)]
        action = np.array([0, 0, 3, 1, 0, 0, 1])
        _, _, done, truncated, info = env.step(action)
        self.assertFalse(done or truncated)
        self.assertEqual((info["waves_completed"], info["kills"], info["damage_taken"]), (1, 1, 3))
        self.assertEqual(env.state.health, 17)  # no between-wave healing/reset
        self.assertEqual([a["attack"] for a in backend.actions[-2:]], [1, 0])
        self.assertEqual([a["camera_yaw"] for a in backend.actions[-2:]], [10, 0])
        backend.script = [observation(kills=13, dealt=700, taken=110, health=17)]
        _, _, done, truncated, info = env.step(action)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual((info["waves_completed"], info["kills"], info["damage_dealt"]), (2, 3, 60))
        self.assertTrue(info["success"])

    def test_death_precedes_wave_success_and_timeout_is_truncation(self):
        env, backend = self.make_env()
        env.settings["max_episode_steps"] = 1
        env.reset(seed=7)
        backend.script = [observation(kills=11, dealt=300, taken=280, health=0)]
        _, _, done, truncated, info = env.step(NO_OP_ACTION)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info["terminal_reason"], "death")
        self.assertEqual(info["waves_completed"], 0)
        env.reset(seed=7)
        _, _, done, truncated, info = env.step(NO_OP_ACTION)
        self.assertFalse(done)
        self.assertTrue(truncated)
        self.assertEqual(info["terminal_reason"], "time_limit")


if __name__ == "__main__":
    unittest.main()
