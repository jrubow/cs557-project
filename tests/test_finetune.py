"""Verify warm-start identity, effective PPO settings, and held-out boundaries."""
import copy
from pathlib import Path
import tempfile
import unittest

from combat_rl.config import load_config
from combat_rl.evaluate import scenario_start
from combat_rl.train import finetune_changes, finetune_ppo_kwargs
from scripts.queue_survival_windows import recipe_config


class FineTuneTests(unittest.TestCase):
    def test_task_changes_are_rejected(self):
        original = load_config()
        for section, key, value in [('environment', 'weapon', 'diamond_sword'),
                                    ('environment', 'action_repeat', 8),
                                    ('evaluation', 'test_seed', 51000),
                                    ('training', 'policy_layers', [256, 256]),
                                    ('training', 'initial_action_bias', 10)]:
            changed = copy.deepcopy(original)
            changed[section][key] = value
            with self.assertRaisesRegex(ValueError, 'cannot change'):
                finetune_changes(original, changed)
        for invalid in (-1, float('nan')):
            changed = copy.deepcopy(original)
            changed['training']['learning_rate'] = invalid
            with self.assertRaises(ValueError):
                finetune_changes(original, changed)

    def test_fresh_ranges_do_not_cross_into_test(self):
        config = load_config()
        self.assertEqual(scenario_start(config, 'validation', 50), 50000)
        self.assertEqual(scenario_start(config, 'validation', 50, 51000), 51000)
        self.assertEqual(scenario_start(config, 'test', 50, 200000), 200000)
        for split, first in [('validation', 49999), ('validation', 99951),
                             ('validation', 100000), ('test', 51000)]:
            with self.assertRaises(ValueError):
                scenario_start(config, split, 50, first)

    def test_loaded_optimizer_and_weights_survive_changed_schedules(self):
        import gymnasium as gym
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        env = gym.make('CartPole-v1')
        try:
            with tempfile.TemporaryDirectory() as directory:
                model = PPO('MlpPolicy', env, n_steps=16, batch_size=8, n_epochs=1,
                            seed=0, learning_rate=0.0003, gamma=0.99, verbose=0)
                model.learn(32)
                path = Path(directory) / 'parent.zip'
                model.save(path)
                config = recipe_config(load_config(), 'survival10')
                loaded = PPO.load(path, env=env, **finetune_ppo_kwargs(config))
                self.assertEqual(loaded.num_timesteps, 32)
                self.assertEqual(loaded.gamma, 0.995)
                self.assertEqual(loaded.rollout_buffer.gamma, 0.995)
                self.assertEqual(loaded.lr_schedule(1), 0.0001)
                self.assertEqual(loaded.clip_range(1), config['training']['clip_range'])
                for key, value in model.policy.state_dict().items():
                    self.assertTrue(torch.equal(value, loaded.policy.state_dict()[key]))
                for key, value in model.policy.optimizer.state_dict()['state'].items():
                    for name, tensor in value.items():
                        self.assertTrue(torch.equal(tensor, loaded.policy.optimizer.state_dict()['state'][key][name]))
                loaded.learn(16, reset_num_timesteps=False)
                self.assertEqual(loaded.num_timesteps, 48)
                self.assertEqual(loaded.policy.optimizer.param_groups[0]['lr'], 0.0001)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
