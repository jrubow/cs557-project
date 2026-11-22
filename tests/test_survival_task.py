"""Real PPO save/load/transfer integration with a simulator-free task backend."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from combat_rl.config import load_config
from combat_rl.endless import EndlessCombatEnv
from combat_rl import survival_task as task
from combat_rl.train import save_checkpoint, read_checkpoint
from scripts.queue_optimization_windows import sha256
from test_env import Backend


class TelemetryBackend(Backend):
    def step(self, action):
        result = super().step(action)
        full = result[0]['full']
        full.sound_subtitles, full.is_on_ground, full.velocity_y = [], True, 0.0
        return result


def environment(config, condition, spec, port=8000):
    return EndlessCombatEnv(config, condition=condition, backend=TelemetryBackend(), **spec)


class SurvivalTaskTests(unittest.TestCase):
    def test_task_spec_rejects_hidden_changes(self):
        for spec in ({'mode':'waves','max_seconds':20}, {'mode':'batch','armor':'diamond'},
                     {'mode':'batch','batch_size':6}, {'mode':'timed','spawn_seconds':.001}):
            with self.assertRaises(ValueError):
                task.validate_spec(spec)

    def test_real_ppo_transfer_continuation_and_telemetry_evaluation_preserve_parent(self):
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        config = load_config()
        config['training'].update(n_steps=8, batch_size=4, n_epochs=1, total_timesteps=16)
        spec = dict(mode='batch', batch_size=2, max_seconds=.4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = environment(config, 'C', spec)
            model = PPO('MlpPolicy', env, n_steps=8, batch_size=4, n_epochs=1, seed=0)
            model.learn(16)
            # Old task episode summaries omit survival-specific keys.
            model.ep_info_buffer.append(dict(r=0,l=2,t=1))
            save_checkpoint(model, root/'parent', dict(config=config,condition='C',seed=0,runtime={}))
            parent, previous = read_checkpoint(root/'parent')
            original_hash = sha256(parent)
            args = argparse.Namespace(output=root/'transfer', additional_steps=17, port=8000)
            with patch.object(task, 'make_environment', side_effect=environment), \
                 patch.object(task, 'runtime_metadata', return_value=dict(git_revision='test',git_status='')), \
                 contextlib.redirect_stdout(io.StringIO()):
                task.train(args, parent, previous, spec)
                current, record = read_checkpoint(args.output/'final')
                self.assertEqual(record['timesteps'], 40)
                self.assertEqual(record['task_start_timesteps'], 16)
                self.assertEqual(record['resume_sha256'], original_hash)
                self.assertEqual(record['task_spec'], spec)
                self.assertEqual(record['config']['training']['learning_rate'], .0001)
                args.output = root/'continuation'
                task.train(args, current, record, spec)
                final, final_record = read_checkpoint(args.output/'final')
                self.assertEqual(final_record['timesteps'], 64)
                self.assertEqual(final_record['task_start_timesteps'], 16)
                evaluation = argparse.Namespace(output=root/'evaluation.csv', episodes=2,first_seed=52100,port=8000)
                with patch.object(EndlessCombatEnv, '_make_backend', side_effect=TelemetryBackend):
                    task.evaluate(evaluation, final, final_record, spec)
                meta = json.loads(evaluation.output.with_suffix('.json').read_text())
                self.assertEqual(meta['training_task'], spec)
                self.assertFalse(meta['policy_updated'])
                self.assertEqual(meta['model_sha256'], sha256(final))
                telemetry = json.loads(evaluation.output.with_suffix('.events.json').read_text())
                self.assertEqual([r['scenario_seed'] for r in telemetry], [52100,52101])
                self.assertEqual(telemetry[0]['summary']['ticks'], 8)
                args.output = root/'wrong-task'
                with self.assertRaisesRegex(ValueError, 'preserve'):
                    task.train(args, final, final_record, dict(spec,batch_size=1))
            self.assertEqual(sha256(parent), original_hash)
            env.close()


if __name__ == '__main__':
    unittest.main()
