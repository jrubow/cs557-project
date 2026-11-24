"""Protect shared-runtime handoff, selection rules and evidence completeness."""
import argparse
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import queue_endless_windows as q


class WaitingWorker(q.Worker):
    def __init__(self, root, alive):
        self.root, self.predecessor, self.control = root/'output', root/'previous', root/'control'
        self.args = argparse.Namespace(predecessor_pid=123)
        self.alive = iter(alive)
        self.state = {}
    def predecessor_alive(self):
        return next(self.alive)
    def check_source(self):
        pass


class EndlessQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def predecessor(self, phase='complete', control='complete', clean=True):
        q.write_json(self.root/'previous/status.json', dict(phase=phase))
        q.write_json(self.root/'control/status.json', dict(phase=control))
        (self.root/'previous/ledger.jsonl').write_text('{"event":"job_started","label":"last"}\n')
        q.write_json(self.root/'previous/logs/last.status.json',
                     dict(exit_code=0, remaining_descendant_pids=[] if clean else [123]))
    def test_waits_for_actual_exit_even_when_status_complete(self):
        self.predecessor()
        worker = WaitingWorker(self.root, [True, False])
        with patch.object(q.time, 'sleep') as sleep:
            worker.wait()
        sleep.assert_called_once_with(15)
        self.assertEqual(q.read_json(worker.root/'status.json')['phase'], 'waiting_for_survival')
    def test_failed_budget_stopped_or_stale_control_cannot_launch(self):
        for phase, control in [('failed','complete'),('budget_stopped','complete'),('complete','failed'),('running','complete')]:
            self.predecessor(phase, control)
            with self.assertRaisesRegex(RuntimeError, 'both successful'):
                WaitingWorker(self.root, [False]).wait()
        self.predecessor(clean=False)
        with self.assertRaisesRegex(RuntimeError, 'unclean'):
            WaitingWorker(self.root, [False]).wait()
    def test_selection_requires_kills_not_just_waiting_longer(self):
        scores = {f'batch{k}':dict(median_seconds=80,median_kills=12) for k in (1,2,3,5)}
        scores['batch5']['median_seconds'] = 40
        self.assertEqual(q.select_batch(scores), 'batch3')
        baseline = dict(mean_kills=10,mean_seconds=70)
        self.assertFalse(q.improves(dict(mean_kills=10,mean_seconds=200), baseline))
        self.assertFalse(q.improves(dict(mean_kills=20,mean_seconds=60), baseline))
        self.assertTrue(q.improves(dict(mean_kills=12,mean_seconds=80), baseline))
    def test_summary_rejects_truncated_scenarios_changed_models_and_test(self):
        model = self.root/'model.zip'
        model.write_bytes(b'checkpoint')
        q.write_json(model.with_suffix('.json'), dict(config={}))
        path = self.root/'eval.csv'
        spec = q.PROFILES['batch1']
        meta = dict(model_sha256=q.sha256(model), evaluation_task=spec, split='validation', first_seed=52100,
                    episodes=2, deterministic=True, policy_updated=False, model=str(model), config={},
                    runtime=dict(git_revision='revision', git_status=''))
        q.write_json(path.with_suffix('.json'), meta)
        row = dict(scenario_seed=52100, mode='batch', model=str(model), success=False, kills=2,
                   elapsed_seconds=30, game_ticks=600, attack_pulses=100, critical_sound_events=1,
                   damage_dealt=40,damage_taken=20,waves_completed=0,batches_completed=2,terminal_reason='death')
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
            writer.writerow(dict(row, scenario_seed=52101))
        score = q.summarize(path, model, spec, 52100, 2, 'revision')
        self.assertEqual(score['critical_cues_per_100_attack_pulses'], 1)
        self.assertEqual(score['critical_cues_per_game_minute'], 2)
        with self.assertRaisesRegex(RuntimeError, 'provenance'):
            q.summarize(path, model, spec, 100000, 2, 'revision')
        model.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'provenance'):
            q.summarize(path, model, spec, 52100, 2, 'revision')
        model.write_bytes(b'checkpoint')
        path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            q.summarize(path, model, spec, 52100, 2, 'revision')


if __name__ == '__main__':
    unittest.main()
