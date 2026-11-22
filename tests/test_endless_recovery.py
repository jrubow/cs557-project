"""Completed artifact adoption must not accept altered or unfinished training."""
import copy
from pathlib import Path
import tempfile
import unittest

from scripts.finish_endless_windows import inspect_final
from scripts.queue_optimization_windows import read_json,write_json,sha256
from test_optimization_queue import checkpoint
from combat_rl.config import load_config


class RecoveryTests(unittest.TestCase):
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
