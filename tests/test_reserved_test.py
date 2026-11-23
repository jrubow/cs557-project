"""Reserved evaluation keeps recipe lineages separate and rejects changed budgets."""
from pathlib import Path
import tempfile
import unittest
from scripts import run_reserved_test_windows as q
from test_optimization_queue import checkpoint,CONFIG


class ReservedTests(unittest.TestCase):
    def test_checkpoint_seed_and_budget_must_match_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.zip'
            checkpoint(path,'C',0,CONFIG,4000000,'source')
            record=q.checkpoint_job('candidate','survival10 4M',path,0,4000000)
            self.assertEqual(record['model_sha256'],q.sha256(path))
            for seed,target in [(1,4000000),(0,5000000)]:
                with self.assertRaisesRegex(RuntimeError,'mismatch'):
                    q.checkpoint_job('candidate','survival10 4M',path,seed,target)

    def test_macro_means_and_paired_seed_changes_keep_batch_exploratory(self):
        rows=[]
        for group,values in [('C 2M parent',[2,3,4]),('survival10 4M',[3,4,5]),('batch +1M transfer',[5])]:
            for seed,value in enumerate(values):
                row=dict(group=group,training_seed=seed,episodes=50,successes=0)
                row.update({k:value for k in q.METRICS})
                rows.append(row)
        result=q.summarize(rows)
        self.assertTrue(result['complete'])
        self.assertEqual(result['episodes'],350)
        self.assertEqual([g['training_seeds'] for g in result['groups']],[3,3,1])
        self.assertEqual(result['groups'][0]['waves_completed_mean'],3)
        self.assertIsNone(result['groups'][2]['waves_completed_sd'])
        self.assertEqual([r['waves_completed'] for r in result['paired_seed_changes']],[1,1,1])
        self.assertFalse(result['selection_changed'])


if __name__=='__main__':
    unittest.main()
