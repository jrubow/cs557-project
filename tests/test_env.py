"""Exercise task boundaries without launching Minecraft or a training job."""

from types import SimpleNamespace
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
