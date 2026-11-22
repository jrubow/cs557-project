"""Verify safe predecessor handoff and truthful winning-video selection."""
import argparse
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import queue_demos_windows as q


class WaitingWorker(q.Worker):
    def __init__(self, root, alive):
        self.root = root/'output'
        self.predecessor = root/'predecessor'
        self.args = argparse.Namespace(wait_hours=1, predecessor_pid=123)
        self.alive = iter(alive)
        self.states = []

    def predecessor_alive(self):
        return next(self.alive)

    def source_check(self):
        pass

    def status(self, phase, **extra):
        self.states.append(phase)


class DemoQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def predecessor(self, phase='complete', clean=True):
        directory = self.root/'predecessor'
        q.write_json(directory/'status.json', dict(phase=phase))
        (directory/'ledger.jsonl').write_text('{"event":"job_started","label":"last"}\n')
        q.write_json(directory/'logs/last.status.json',
                     dict(exit_code=0, remaining_descendant_pids=[] if clean else [1234]))

    def test_terminal_status_alone_does_not_allow_concurrent_simulator(self):
        self.predecessor()
        worker = WaitingWorker(self.root, [True, False])
        with patch.object(q.time, 'sleep') as sleep:
            result = worker.wait_for_predecessor()
        self.assertEqual(result['phase'], 'complete')
        sleep.assert_called_once_with(15)

    def test_dead_predecessor_or_unclean_child_cannot_start_demos(self):
        self.predecessor(phase='running')
        with self.assertRaisesRegex(RuntimeError, 'without a terminal'):
            WaitingWorker(self.root, [False]).wait_for_predecessor()
        self.predecessor(clean=False)
        with self.assertRaisesRegex(RuntimeError, 'unclean'):
            WaitingWorker(self.root, [False]).wait_for_predecessor()

    def test_zero_wait_budget_waits_for_actual_exit(self):
        self.predecessor()
        worker = WaitingWorker(self.root, [True, True, False])
        worker.args.wait_hours = 0
        with patch.object(q.time, 'sleep') as sleep:
            self.assertEqual(worker.wait_for_predecessor()['phase'], 'complete')
        self.assertEqual(sleep.call_count, 2)

    def test_seed_order_uses_only_matching_development_results(self):
        directory = self.root/'evaluations'
        directory.mkdir()
        model = self.root/'policy.zip'
        meta = dict(model=str(model), split='validation', first_seed=51000, episodes=50, deterministic=True)
        q.write_json(directory/'development.json', meta)
        with (directory/'development.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['scenario_seed', 'success', 'waves_completed', 'terminal_reason'])
            writer.writeheader()
            for seed in q.DEV_SEEDS:
                win = seed == 51022
                writer.writerow(dict(scenario_seed=seed, success=win, waves_completed=5 if win else 3,
                                     terminal_reason='success' if win else 'death'))
        q.write_json(directory/'test.json', dict(meta, split='test', first_seed=200000))
        # No test.csv exists: accidentally reading it would fail this test.
        order = q.winning_seed_order(self.root, model)
        self.assertEqual(order[0], 51022)
        self.assertEqual(sorted(order), q.DEV_SEEDS)

    def test_video_must_contain_a_complete_win_and_correct_frames(self):
        directory = self.root/'video'
        record = dict(model_sha256='hash', scenario_seed=51000, deterministic=True,
                      metrics=dict(success=True, waves_completed=5, terminal_reason='success', kills=15, elapsed_seconds=1),
                      raw_frames=21, annotated_frames=61)
        q.write_json(directory/'episode.json', record)
        decoder = SimpleNamespace(count_frames_and_secs=Mock(side_effect=[(21, 1.05), (61, 3.05)]))
        with patch.dict('sys.modules', imageio_ffmpeg=decoder):
            q.verify_video(directory, 'hash', require_win=True)
        decoder = SimpleNamespace(count_frames_and_secs=Mock(return_value=(1, 0.05)))
        with patch.dict('sys.modules', imageio_ffmpeg=decoder):
            with self.assertRaisesRegex(RuntimeError, 'frames/duration'):
                q.verify_video(directory, 'hash', require_win=True)
        record['metrics']['waves_completed'] = 4
        q.write_json(directory/'episode.json', record)
        with self.assertRaisesRegex(RuntimeError, 'not a complete'):
            q.verify_video(directory, 'hash', require_win=True)


if __name__ == '__main__':
    unittest.main()
