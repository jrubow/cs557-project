"""Check continuous spawning on exact ticks without starting Minecraft."""
import copy
import unittest

import numpy as np

from combat_rl.config import load_config
from combat_rl.endless import EndlessCombatEnv
from combat_rl.env import ENTITY_RANGE
from combat_rl.task import NO_OP_ACTION
from test_env import Backend, observation


class AccumulatingBackend(Backend):
    def __init__(self):
        super().__init__()
        self.spawn_ticks = []

    def add_commands(self, commands):
        self.commands.extend(commands)
        count = sum(command.startswith('summon minecraft:zombie ') for command in commands)
        if count:
            self.spawn_ticks.append(len(self.actions)+1)
            state = self.current['full']
            self.current = observation(len(state.surrounding_entities[ENTITY_RANGE].entities)+count,
                kills=state.killed_statistics['zombie'], dealt=state.misc_statistics['damage_dealt'],
                taken=state.misc_statistics['damage_taken'], health=state.health)


class EndlessTests(unittest.TestCase):
    def make_env(self, seconds=12):
        config = load_config()
        config['environment']['max_episode_steps'] = 1  # Demo owns its recording limit.
        backend = AccumulatingBackend()
        env = EndlessCombatEnv(config, backend=backend, max_seconds=seconds)
        obs, info = env.reset(seed=51000)
        backend.actions.clear()
        backend.spawn_ticks.clear()
        return env, backend, obs, info

    def test_cadence_can_accumulate_more_than_five_without_expanding_observation(self):
        env, backend, obs, info = self.make_env()
        self.assertEqual(obs.shape, (156,))
        self.assertEqual(info['spawn_commands_issued'], 1)
        self.assertEqual(float(obs[6]), 1.0)
        for _ in range(60):
            obs, _, terminated, truncated, info = env.step(NO_OP_ACTION)
        self.assertEqual(backend.spawn_ticks, [40, 80, 120, 160, 200, 240])
        self.assertEqual(info['active_enemies'], 7)
        self.assertEqual(info['peak_active_enemies'], 7)
        self.assertEqual(info['spawn_commands_issued'], 7)
        self.assertEqual(info['elapsed_seconds'], 12)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertFalse(terminated or info['success'])
        self.assertTrue(truncated)
        self.assertEqual(info['terminal_reason'], 'recording_time_limit')
        self.assertEqual(info['waves_completed'], 0)

    def test_killing_every_enemy_does_not_end_heal_or_start_a_batch_wave(self):
        env, backend, _, _ = self.make_env()
        config_before = copy.deepcopy(env.config)
        backend.script = [observation(count=0, kills=11, dealt=300, taken=110, health=17)]
        _, _, terminated, truncated, info = env.step(NO_OP_ACTION)
        self.assertFalse(terminated or truncated)
        self.assertEqual((info['kills'], info['final_health'], info['spawn_commands_issued']), (1, 17, 1))
        self.assertEqual(backend.spawn_ticks, [])
        self.assertEqual(info['damage_dealt'], 20)
        self.assertEqual(info['damage_taken'], 3)
        for _ in range(9):
            _, _, _, _, info = env.step(NO_OP_ACTION)
        self.assertEqual(info['final_health'], 17)
        self.assertEqual(backend.spawn_ticks, [40])
        self.assertEqual(env.config, config_before)

    def test_death_priority_action_pulses_and_exact_partial_decision_limit(self):
        env, backend, _, _ = self.make_env(seconds=0.15)
        action = np.array([0, 0, 3, 1, 0, 0, 1])
        _, _, terminated, truncated, info = env.step(action)
        self.assertEqual(len(backend.actions), 3)
        self.assertEqual([a['attack'] for a in backend.actions], [1, 0, 0])
        self.assertEqual([a['camera_yaw'] for a in backend.actions], [10, 0, 0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        env.reset(seed=51000)
        backend.script = [observation(health=0)]
        _, _, terminated, truncated, info = env.step(NO_OP_ACTION)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['terminal_reason'], 'death')
        self.assertFalse(info['success'])

    def test_invalid_timing_is_rejected(self):
        for settings in [dict(spawn_seconds=0), dict(spawn_seconds=float('nan')),
                         dict(spawn_seconds=0.03), dict(max_seconds=601)]:
            with self.assertRaises(ValueError):
                EndlessCombatEnv(load_config(), **settings)

    def test_batch_waits_for_every_kill_then_respawns_without_healing(self):
        backend = AccumulatingBackend()
        env = EndlessCombatEnv(load_config(), backend=backend, mode='batch', batch_size=3)
        obs, info = env.reset(seed=52000)
        self.assertEqual(info['active_enemies'], 3)
        self.assertAlmostEqual(float(obs[6]), .6)
        backend.commands.clear()
        backend.script = [observation(1, kills=12, health=8)]
        _, _, done, cut, info = env.step(NO_OP_ACTION)
        self.assertEqual(info['spawn_commands_issued'], 3)
        self.assertEqual(info['batches_completed'], 0)
        backend.script = [observation(0, kills=13, health=8)]
        _, _, done, cut, info = env.step(NO_OP_ACTION)
        self.assertFalse(done or cut)
        self.assertEqual(info['spawn_commands_issued'], 6)
        self.assertEqual(info['batches_completed'], 1)
        self.assertEqual(info['kills'], 3)
        self.assertEqual(info['final_health'], 8)
        self.assertEqual(info['active_enemies'], 3)
        self.assertEqual(info['peak_active_enemies'], 3)
        self.assertTrue(all(command.startswith('summon minecraft:zombie ') for command in backend.commands))
        for _ in range(20):
            env.step(NO_OP_ACTION)
        self.assertEqual(env.spawn_commands_issued, 6)  # Timed arrivals are disabled.

    def test_empty_or_dead_observation_cannot_double_spawn_a_batch(self):
        for health, seconds in [(0,300), (7,.05)]:
            backend = AccumulatingBackend()
            env = EndlessCombatEnv(load_config(), backend=backend, mode='batch', batch_size=2, max_seconds=seconds)
            env.reset(seed=52000)
            backend.script = [observation(0, kills=12, health=health)]
            _, _, done, cut, info = env.step(NO_OP_ACTION)
            self.assertEqual(info['spawn_commands_issued'], 2)
            self.assertEqual(info['batches_completed'], int(health > 0))
            self.assertEqual(done, health == 0)
            self.assertEqual(cut, health > 0)
        backend = AccumulatingBackend()
        env = EndlessCombatEnv(load_config(), backend=backend, mode='batch', batch_size=2)
        env.reset(seed=52000)
        backend.script = [observation(0, kills=10)]
        env.step(NO_OP_ACTION)
        self.assertEqual(env.batches_completed, 0)  # Empty entity list alone is insufficient.

    def test_invalid_batch_size_and_mode_rejected(self):
        for kwargs in (dict(mode='other'),dict(batch_size=0),dict(batch_size=6),dict(batch_size=2.5)):
            with self.assertRaises(ValueError):
                EndlessCombatEnv(load_config(), **kwargs)


if __name__ == '__main__':
    unittest.main()
