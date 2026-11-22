"""Reuse existing jobs without launching Python or repeating training."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import continue_survival_windows as c
from test_optimization_queue import checkpoint, CONFIG

q = c.q


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.worker = object.__new__(c.ContinuationWorker)
        w = self.worker
        w.root, w.control = self.root/'run', self.root/'control'
        w.source, w.revision = self.root/'source', 'source'
        w.controller_source, w.controller_revision = self.root/'controller', 'controller'
        w.args = argparse.Namespace(handoff_token='authorized', predecessor_pid=123)
        w.source_check = Mock()
        w.ledger = Mock()
        w.status = Mock()
        w.rate, w.jobs, w.state = 0.005, 0, {}
        w.started = c.time.monotonic()
        self.parent = self.root/'parent/final.zip'
        checkpoint(self.parent, 'C', 0, CONFIG, 2_000_000, 'source')
        prior = q.read_json(self.parent.with_suffix('.json'))
        self.config = q.recipe_config(prior['config'], 'survival10')
        self.config['training']['total_timesteps'] = 2_500_000
        self.name = 'survival10_2500000_seed0'
        self.model = w.root/'models'/self.name/'final.zip'
        checkpoint(self.model, 'C', 0, self.config, 2_500_000, 'source', self.parent)
        meta = q.read_json(self.model.with_suffix('.json'))
        meta['resume_sha256'] = q.sha256(self.parent)
        q.write_json(self.model.with_suffix('.json'), meta)
        self.frozen = w.root/'configs'/(self.name+'.json')
        q.write_json(self.frozen, self.config)
        self.arguments = ['-m', 'combat_rl.train', '--run-dir', str(self.model.parent),
                          '--resume', str(self.parent), '--steps', '2500000', '--finetune-config', str(self.frozen)]
        q.write_json(w.root/'logs'/(self.name+'-train.status.json'), dict(exit_code=0,
            remaining_descendant_pids=[], wall_seconds=2500, command=['python', *self.arguments]))
        w.completed_events = [dict(event='job_completed', label=self.name+'-train'),
            dict(event='checkpoint', model=str(self.model), model_sha256=q.sha256(self.model),
                 metadata_sha256=q.sha256(self.model.with_suffix('.json')))]

    def tearDown(self):
        self.temp.cleanup()

    def test_finished_training_is_reused_without_a_subprocess(self):
        before = q.sha256(self.model)
        with patch.object(q.subprocess, 'Popen') as process:
            result = self.worker.train('survival10', 0, 2_500_000, str(self.parent), first=True)
        process.assert_not_called()
        self.assertEqual(result, str(self.model))
        self.assertEqual(q.sha256(self.model), before)
        self.assertEqual(self.worker.jobs, 1)

    def test_changed_checkpoint_or_arguments_cannot_be_silently_reused(self):
        with self.assertRaisesRegex(RuntimeError, 'arguments changed'):
            self.worker.job(self.name+'-train', self.arguments+['--device', 'cpu'], 0)
        changed = q.read_json(self.model.with_suffix('.json'))
        changed['resume_sha256'] = 'changed'
        q.write_json(self.model.with_suffix('.json'), changed)
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            self.worker.train('survival10', 0, 2_500_000, str(self.parent), first=True)

    def test_no_overall_deadline_and_status_serializes_null(self):
        self.assertEqual(self.worker.remaining(), float('inf'))
        c.ContinuationWorker.status(self.worker, 'running', current_job='example')
        status = q.read_json(self.worker.root/'status.json')
        self.assertIsNone(status['remaining_seconds'])
        self.assertTrue(status['overall_time_limit_removed'])

    def prepare_handoff(self, token='authorized'):
        w = self.worker
        w.parent_models = []
        w.parents = []
        originals = []
        for seed in range(3):
            model = self.root/f'parents/seed{seed}/final.zip'
            checkpoint(model, 'C', seed, CONFIG, 2_000_000, 'source')
            w.parent_models.append(model)
            w.parents.append(q.read_json(model.with_suffix('.json')))
            originals.append(dict(model=str(model), model_sha256=q.sha256(model),
                                  metadata_sha256=q.sha256(model.with_suffix('.json'))))
        q.write_json(w.root/'manifest.json', dict(source=str(w.source), source_revision='source',
            development_scenarios=list(range(51000, 51050)), reserved_test_scenarios=list(range(200000, 200050)),
            recipes={r: q.recipe_config(w.parents[0]['config'], r) for r in q.RECIPES},
            parents=originals, parent_config=w.parents[0]['config'], max_hours=18))
        w.original_manifest_hash = q.sha256(w.root/'manifest.json')
        q.write_json(w.root/'status.json', dict(phase='budget_stopped'))
        (w.root/'ledger.jsonl').write_text('')
        q.write_json(w.root/'STOP_AFTER_CURRENT', dict(handoff_token=token))
        w.predecessor_alive = lambda: False
        w.predecessor_identity = None
        return w

    def test_handoff_does_not_remove_an_unrelated_stop_request(self):
        w = self.prepare_handoff(token='another-request')
        with self.assertRaisesRegex(RuntimeError, 'unrelated stop'):
            w.initialize()
        self.assertTrue((w.root/'STOP_AFTER_CURRENT').exists())
        self.assertEqual(q.sha256(w.root/'manifest.json'), w.original_manifest_hash)

    def test_successful_handoff_writes_real_ledger_and_status_without_timestamp_collision(self):
        import json
        w = self.prepare_handoff()
        # Exercise real persistence, which mocking ledger/status would miss.
        w.ledger = q.Worker.ledger.__get__(w, c.ContinuationWorker)
        w.status = c.ContinuationWorker.status.__get__(w, c.ContinuationWorker)
        w.initialize()
        self.assertTrue(w.ready)
        self.assertFalse((w.root/'STOP_AFTER_CURRENT').exists())
        self.assertEqual(q.sha256(w.root/'manifest.json'), w.original_manifest_hash)
        record = json.loads((w.root/'ledger.jsonl').read_text().splitlines()[-1])
        self.assertEqual(record['event'], 'controller_handoff')
        self.assertIn('utc', record)
        self.assertIn('utc', record['amendment'])
        status = q.read_json(w.root/'status.json')
        self.assertEqual(status['phase'], 'resuming_completed_protocol')
        self.assertIsNone(status['remaining_seconds'])


if __name__ == '__main__':
    unittest.main()
