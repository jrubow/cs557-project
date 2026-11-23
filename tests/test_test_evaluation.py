"""Check held-out result identity and separation of unequal training budgets."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest

from scripts import run_test_evaluation_windows as test_queue
from test_optimization_queue import CONFIG, checkpoint, evaluation


class HeldOutEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "model/final.zip"
        checkpoint(self.model, "C", 0, CONFIG, 1000000, "source")
        self.csv = self.root / "test.csv"
        evaluation(self.csv, self.model, [2, 3])
        self.job = dict(name="C_1M_seed0", group="PPO C 1M", policy="ppo", condition="C",
                        training_seed=0, training_target=1000000, model=str(self.model))
        self.meta = test_queue.read_json(self.csv.with_suffix(".json"))
        with self.csv.open(newline="") as stream:
            self.rows = list(csv.DictReader(stream))

    def tearDown(self):
        self.temp.cleanup()

    def make_test(self):
        self.meta.update(split="test", first_seed=100000)
        for index, row in enumerate(self.rows):
            row.update(split="test", scenario_seed=str(100000 + index))
        self.save()

    def save(self):
        test_queue.write_json(self.csv.with_suffix(".json"), self.meta)
        with self.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def test_validation_results_cannot_be_reported_as_test(self):
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            test_queue.validate_result(self.csv, self.job, CONFIG)

    def test_complete_test_passes_but_duplicate_scenarios_fail(self):
        self.make_test()
        result = test_queue.validate_result(self.csv, self.job, CONFIG)
        self.assertEqual(result["waves_completed"], 2.5)
        self.rows[-1]["scenario_seed"] = self.rows[0]["scenario_seed"]
        self.save()
        with self.assertRaisesRegex(RuntimeError, "scenarios"):
            test_queue.validate_result(self.csv, self.job, CONFIG)

    def test_changed_model_identity_is_rejected(self):
        self.make_test()
        self.rows[0]["training_seed"] = "1"
        self.save()
        with self.assertRaisesRegex(RuntimeError, "identity"):
            test_queue.validate_result(self.csv, self.job, CONFIG)

    def test_reserved_range_is_explicit_and_cannot_accept_original_test(self):
        self.make_test()
        with self.assertRaisesRegex(RuntimeError,'metadata'):
            test_queue.validate_result(self.csv,self.job,CONFIG,200000)
        self.meta['first_seed']=200000
        for index,row in enumerate(self.rows):
            row['scenario_seed']=str(200000+index)
        self.save()
        self.assertEqual(test_queue.validate_result(self.csv,self.job,CONFIG,200000)['episodes'],50)
        with self.assertRaisesRegex(RuntimeError,'metadata'):
            test_queue.validate_result(self.csv,self.job,CONFIG)

    def test_training_budgets_remain_separate_groups(self):
        self.make_test()
        first = test_queue.validate_result(self.csv, self.job, CONFIG)
        second = copy.deepcopy(first)
        second.update(name="C_2M_seed0", group="PPO C 2M", training_target=2000000)
        groups = test_queue.summarize(self.root, [first, second])
        self.assertEqual([g["group"] for g in groups], ["PPO C 1M", "PPO C 2M"])
        self.assertTrue(all(g["independent_training_runs"] == 1 and g["waves_completed_sd"] is None for g in groups))


if __name__ == "__main__":
    unittest.main()
