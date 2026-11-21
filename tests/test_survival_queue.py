"""Exercise complete bounded selection/confirmation paths without Minecraft."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest

from scripts import queue_survival_windows as q
from test_optimization_queue import checkpoint, CONFIG


def score(waves, success=0):
    return dict(waves=waves, success_rate=success, kills=waves*2,
                damage_dealt=waves*40, damage_taken=20)


class SimulatedWorker(q.Worker):
    def __init__(self, args, regress=False, fail_confirmation=False):
        super().__init__(args)
        self.calls = []
        self.regress = regress
        self.fail_confirmation = fail_confirmation

    def git(self, *args):
        return '' if args[0] == 'status' else 'source'

    def source_check(self):
        pass

    def train(self, recipe, seed, target, parent, first=False, reserve=0):
        self.calls.append((recipe, seed, target, first, str(parent)))
        path = self.root / 'models' / f'{recipe}_{target}_seed{seed}/final.zip'
        checkpoint(path, 'C', seed, CONFIG, target, 'source', parent)
        return str(path)

    def evaluate(self, name, model):
        if name.startswith('parent'):
            return score(2.8)
        if self.regress or (self.fail_confirmation and name.startswith('confirmed')):
            return score(1)
        if name.startswith('confirmed') or name == 'selected_seed0_repeat':
            return score(4.2, 0.2)
        target = int(name.split('_')[1])
        recipe = name.split('_')[0]
        if target == 5_000_000:
            return score(4.2, 0.2)
        if target == 4_000_000:
            return score(3.8, 0.1)
        return score({'continue': 2.9, 'survival10': 3.4, 'survival25': 3.1}[recipe])

    def record(self, model):
        self.recorded = str(model)


class SurvivalQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = argparse.Namespace(parents=self.root/'parents', output=self.root/'run',
                                       video_deps=self.root/'deps', max_hours=18)
        for seed in range(3):
            checkpoint(self.args.parents/f'more_2000000_seed{seed}/final.zip', 'C', seed,
                       CONFIG, 2_000_000, 'source')

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_selection_matches_restart_boundaries_for_all_seeds(self):
        worker = SimulatedWorker(self.args)
        worker.run()
        selection = q.read_json(self.args.output/'selection.json')
        self.assertEqual(selection['status'], 'confirmed_improvement')
        self.assertEqual(selection['best_seed0']['recipe'], 'survival10')
        for seed in (1, 2):
            calls = [c for c in worker.calls if c[1] == seed]
            self.assertEqual([c[2] for c in calls], [2_500_000, 3_000_000, 4_000_000, 5_000_000])
            self.assertEqual([c[3] for c in calls], [True, False, False, False])
            self.assertTrue(all(c[0] == 'survival10' for c in calls))
        self.assertFalse(selection['new_test_used'])
        self.assertEqual(selection['reserved_test_scenarios'], [200000, 200049])
        self.assertEqual(len(selection['retained_models']), 3)

    def test_regressions_keep_parents_and_stop_unproductive_recipes(self):
        worker = SimulatedWorker(self.args, regress=True)
        worker.run()
        selection = q.read_json(self.args.output/'selection.json')
        self.assertEqual(selection['status'], 'no_confirmed_gain')
        self.assertEqual(len(worker.calls), 4)  # Preflight plus three stopped trials.
        self.assertEqual(selection['best_seed0']['recipe'], 'parent')
        self.assertEqual(selection['retained_models'], [str(p) for p in worker.parent_models])

    def test_lucky_seed_does_not_replace_confirmed_parent_recipe(self):
        worker = SimulatedWorker(self.args, fail_confirmation=True)
        worker.run()
        selection = q.read_json(self.args.output/'selection.json')
        self.assertEqual(selection['status'], 'confirmation_failed')
        self.assertEqual(selection['retained_models'], [str(p) for p in worker.parent_models])

    def test_recipe_preserves_environment_observations_and_actions(self):
        for recipe in q.RECIPES:
            candidate = q.recipe_config(CONFIG, recipe)
            self.assertEqual(candidate['environment'], CONFIG['environment'])
            self.assertEqual(candidate['evaluation'], CONFIG['evaluation'])
            original = copy.deepcopy(CONFIG)
            q.finetune_changes(original, candidate)
            self.assertEqual(original, CONFIG)

    def test_budget_and_stop_marker_prevent_launch(self):
        worker = SimulatedWorker(self.args)
        self.args.output.mkdir()
        with self.assertRaises(q.BudgetStop):
            worker.job('unused', [], worker.remaining()+1)
        (self.args.output/'STOP_AFTER_CURRENT').touch()
        with self.assertRaises(q.BudgetStop):
            worker.job('unused', [], 1)
        self.assertFalse((self.args.output/'logs').exists())


if __name__ == '__main__':
    unittest.main()
