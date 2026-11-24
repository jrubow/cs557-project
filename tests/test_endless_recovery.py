"""Completed artifact adoption must not accept altered or unfinished training."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import argparse

from scripts.finish_endless_windows import inspect_final
from scripts import queue_endless_windows as q
from scripts.queue_optimization_windows import read_json,write_json,sha256
from test_optimization_queue import checkpoint
from combat_rl.config import load_config


class RecoveryTests(unittest.TestCase):
    def test_first_job_creates_logs_before_capturing_supervisor(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=object.__new__(q.Worker)
            worker.root=Path(tmp)/'fresh'
            worker.source=Path(tmp)
            worker.args=argparse.Namespace(video_deps=Path(tmp))
            worker.state={'jobs_completed':0}
            worker.check_source=Mock()
            worker.predecessor_alive=lambda:False
            worker.status=Mock(); worker.ledger=Mock()
            process=Mock(returncode=0)
            process.poll.return_value=0
            def launch(command,**kwargs):
                log=worker.root/'logs/first.log'
                self.assertTrue(log.parent.is_dir())
                self.assertFalse(kwargs['stdout'].closed)
                write_json(log.with_suffix('.status.json'),dict(exit_code=0,remaining_descendant_pids=[]))
                return process
            socket=Mock()
            socket.__enter__=Mock(return_value=socket)
            socket.__exit__=Mock(return_value=False)
            socket.connect_ex.return_value=1
            with patch.object(q.psutil,'process_iter',return_value=[]), \
                 patch.object(q.socket,'socket',return_value=socket), \
                 patch.object(q.subprocess,'Popen',side_effect=launch):
                worker.job('first',['-m','example'])
            self.assertEqual(worker.state['jobs_completed'],1)
            self.assertTrue((worker.root/'logs/first.supervisor.log').exists())

    def test_final_model_budget_parent_and_task_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=load_config()
            config['training']['learning_rate']=.0001
            task=dict(mode='batch',batch_size=3,max_seconds=300)
            parent,model=root/'parent.zip',root/'final.zip'
            checkpoint(parent,'C',0,config,5_000_000,'original')
            checkpoint(model,'C',0,config,6_000_000,'original',parent)
            meta=read_json(model.with_suffix('.json'))
            meta.update(task_spec=task,previous_task=task,resume_sha256=sha256(parent),task_start_timesteps=4_000_000)
            write_json(model.with_suffix('.json'),meta)
            self.assertEqual(inspect_final(model,parent,task,4000000,'original'),meta)
            for key,value in [('resume_sha256','bad'),('timesteps',5000000),('task_spec',dict(task,batch_size=1))]:
                changed=copy.deepcopy(meta); changed[key]=value
                write_json(model.with_suffix('.json'),changed)
                with self.assertRaisesRegex(RuntimeError,'mismatch'):
                    inspect_final(model,parent,task,4000000,'original')


if __name__=='__main__':
    unittest.main()
