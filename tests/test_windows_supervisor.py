"""Supervisor receipts survive cleanup exceptions and record real child exits."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

from scripts import windows_run as w
from scripts.queue_optimization_windows import read_json,clean_supervisor


class SupervisorTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows process flags')
    def test_real_success_and_failure_have_durable_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for code in (0,7):
                path=Path(tmp)/f'{code}.log'
                record=w.supervise([sys.executable,'-c',f'print("child"); raise SystemExit({code})'],
                                   os.environ.copy(),path,30)
                self.assertEqual(read_json(path.with_suffix('.status.json')),record)
                self.assertEqual(record['exit_code'],code)
                self.assertEqual(record['child_exit_code'],code)
                self.assertEqual(record['remaining_descendant_pids'],[])
                self.assertEqual(record['supervisor_errors'],[])
                self.assertIn('child',path.read_text())

    def test_cleanup_failure_writes_failure_not_missing_or_fake_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            child=Mock(pid=123,returncode=0)
            child.poll.return_value=0
            root=Mock(pid=123)
            root.is_running.side_effect=w.psutil.AccessDenied(123)
            with patch.object(w.subprocess,'Popen',return_value=child), \
                 patch.object(w.psutil,'Process',return_value=root), \
                 patch.object(w.psutil,'wait_procs',return_value=([],[])):
                path=Path(tmp)/'failed.log'
                record=w.supervise(['python','-m','example'],{},path,30)
            self.assertEqual(record['child_exit_code'],0)
            self.assertEqual(record['exit_code'],1)
            self.assertTrue(record['supervisor_errors'])
            with self.assertRaisesRegex(RuntimeError,'unclean'):
                clean_supervisor(path.with_suffix('.status.json'))


if __name__=='__main__':
    unittest.main()
