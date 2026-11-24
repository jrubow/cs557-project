"""Behavior diagnostics wait for successful batch completion and cleanup."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import queue_behavior_windows as q


class WaitingWorker(q.Worker):
    def __init__(self,root,alive):
        self.root,self.predecessor = root/'new',root/'batch'
        self.history_path = self.predecessor/'critical-history.json'
        self.history_hash = q.sha256(self.history_path)
        self.state = {}
        self.args = argparse.Namespace(predecessor_pid=123)
        self.alive = iter(alive)
    def predecessor_alive(self):
        return next(self.alive)
    def check_source(self):
        pass


class BehaviorQueueTests(unittest.TestCase):
    def test_live_or_failed_predecessor_cannot_start_simulator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q.write_json(root/'batch/critical-history.json',{})
            q.write_json(root/'batch/status.json',dict(phase='complete'))
            event = dict(event='job_started',data=dict(label='last',command=['python','supervisor','--','-m','job']))
            (root/'batch/ledger.jsonl').write_text(json.dumps(event)+'\n')
            q.write_json(root/'batch/logs/last.status.json',dict(exit_code=0,remaining_descendant_pids=[],command=['python','-m','job']))
            worker = WaitingWorker(root,[True,False])
            with patch.object(q.time,'sleep') as sleep:
                worker.wait()
            sleep.assert_called_once_with(15)
            self.assertEqual(q.read_json(worker.root/'status.json')['predecessor_phase'],'complete')
            for phase in ('failed','stopped','running'):
                q.write_json(root/'batch/status.json',dict(phase=phase))
                with self.assertRaisesRegex(RuntimeError,'without successful'):
                    WaitingWorker(root,[False]).wait()
            q.write_json(root/'batch/status.json',dict(phase='complete'))
            q.write_json(root/'batch/logs/last.status.json',dict(exit_code=0,remaining_descendant_pids=[99]))
            with self.assertRaisesRegex(RuntimeError,'unclean'):
                WaitingWorker(root,[False]).wait()


if __name__=='__main__':
    unittest.main()
