"""Exercise actual PPO inference and passive tick capture without Minecraft."""
import argparse
import contextlib
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from combat_rl import behavior_eval as b
from combat_rl.config import load_config
from combat_rl.endless import EndlessCombatEnv
from combat_rl.train import save_checkpoint
from scripts.queue_optimization_windows import sha256,write_json
from scripts.continue_behavior_windows import Worker as ContinuationWorker
from test_survival_task import TelemetryBackend,environment


class BehaviorEvaluationTests(unittest.TestCase):
    def test_live_ppo_api_and_references_produce_complete_passive_traces(self):
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        config = load_config()
        spec = dict(mode='batch',batch_size=2,max_seconds=.4)
        env = environment(config,'C',spec)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model = PPO('MlpPolicy',env,n_steps=8,batch_size=4,n_epochs=1,seed=0)
                save_checkpoint(model,root/'parent',dict(config=config,condition='C',seed=0,runtime={}))
                write_json(root/'task.json',spec)
                original = sha256(root/'parent.zip')
                for policy in ('ppo','random','steady'):
                    args = argparse.Namespace(model=root/'parent.zip',task_spec=root/'task.json',
                        output=root/policy,policy=policy,first_seed=53000,episodes=2,port=8000)
                    with patch.object(b,'make_environment',side_effect=environment), \
                         patch.object(EndlessCombatEnv,'_make_backend',side_effect=TelemetryBackend), \
                         patch.object(b,'runtime_metadata',return_value=dict(git_revision='test',git_status='')), \
                         contextlib.redirect_stdout(io.StringIO()):
                        b.evaluate(args)
                    with gzip.open(args.output/'ticks.csv.gz','rt',newline='') as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows),16)
                    self.assertEqual([int(r['tick']) for r in rows[:8]],list(range(1,9)))
                    summary = json.loads((args.output/'summary.json').read_text())
                    meta = json.loads((args.output/'metadata.json').read_text())
                    self.assertEqual(summary['counts']['ticks'],16)
                    self.assertEqual(summary['partial_windows'],2)
                    self.assertEqual(summary['focus_fire']['counts']['ticks'],16)
                    self.assertIsNone(summary['focus_fire']['mean_kill_seconds'])
                    self.assertIn('entity_health_after',rows[0])
                    self.assertFalse(meta['policy_updated'])
                    self.assertEqual(meta['policy'],policy)
                    self.assertEqual(meta['deterministic'],policy!='random')
                    self.assertEqual(sha256(root/'parent.zip'),original)
                    # Exercise real validation/reuse of these actual PPO/reference outputs.
                    worker=object.__new__(ContinuationWorker)
                    worker.models={}; worker.state={'jobs_reused':0}; worker.old_revision='test'
                    worker.status=Mock(); worker.ledger=Mock(); worker.job=Mock()
                    expected=dict(name=policy,model=str((root/'parent.zip').resolve()),steps=meta['model_steps'],
                        policy=policy,task=spec,output=str(args.output),summary=summary,model_sha256=original)
                    worker.cached={policy:dict(result=expected,metadata_sha256=sha256(args.output/'metadata.json'))}
                    self.assertEqual(worker.evaluate_behavior(policy,root/'parent.zip',spec,53000,2,policy),expected)
                    worker.job.assert_not_called()
                    self.assertEqual(worker.state['jobs_reused'],1)
                    worker.cached[policy]['metadata_sha256']='altered'
                    with self.assertRaisesRegex(RuntimeError,'metadata changed'):
                        worker.evaluate_behavior(policy,root/'parent.zip',spec,53000,2,policy)
        finally:
            env.close()


if __name__=='__main__':
    unittest.main()
