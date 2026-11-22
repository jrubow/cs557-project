"""Validate kill clocks, incomplete observations, sweeps and target continuity."""
from types import SimpleNamespace as NS
import unittest

from combat_rl.env import ENTITY_RANGE
from combat_rl.focus import FocusTracker,summarize


def snapshot(health, kills=0):
    return NS(surrounding_entities={ENTITY_RANGE:NS(entities=[
        NS(unique_name=uid,health=hp,translation_key='entity.minecraft.zombie')
        for uid,hp in health.items()])},killed_statistics={'zombie':kills})


class FocusTests(unittest.TestCase):
    def tracker(self, health):
        tracker = FocusTracker()
        tracker.begin(snapshot(health))
        return tracker

    def test_clock_starts_at_damage_not_spawn_and_counts_lingering_corpse_once(self):
        t = self.tracker({'a':20})
        for _ in range(60):
            t.observe(snapshot({'a':20}))
        t.observe(snapshot({'a':12}))  # tick 61
        for _ in range(39):
            t.observe(snapshot({'a':12}))
        t.observe(snapshot({'a':0},1))  # tick 101, exactly two seconds after damage
        for _ in range(20):
            t.observe(snapshot({'a':0},1))
        record = t.finish()
        score = summarize([record])
        self.assertEqual(score['mean_kill_seconds'],2)
        self.assertEqual(score['completed_zombies'],1)
        self.assertEqual(score['health_confirmed_deaths'],1)
        self.assertEqual(score['counts']['player_kills'],1)
        self.assertEqual(record['zombies'][0]['first_damage_tick'],61)
        self.assertEqual(record,t.finish())

    def test_single_tick_lethal_hit_is_zero_duration_not_missing(self):
        t = self.tracker({'a':20})
        t.observe(snapshot({'a':0},1))
        score = summarize([t.finish()])
        self.assertEqual(score['mean_kill_seconds'],0)
        self.assertEqual(score['completed_zombies'],1)

    def test_disappearance_and_aggregate_kill_do_not_prove_which_entity_died(self):
        t = self.tracker({'a':20,'b':20})
        t.observe(snapshot({'a':10,'b':20}))
        t.observe(snapshot({},1))
        score = summarize([t.finish()])
        self.assertIsNone(score['mean_kill_seconds'])
        self.assertEqual(score['lost_damaged_zombies'],1)
        self.assertEqual(score['health_confirmed_deaths'],0)

    def test_episode_end_retains_unfinished_clock(self):
        t = self.tracker({'a':20})
        t.observe(snapshot({'a':10}))
        for _ in range(20):
            t.observe(snapshot({'a':10}))
        record = t.finish()
        score = summarize([record])
        self.assertEqual(score['alive_damaged_at_end'],1)
        self.assertEqual(record['zombies'][0]['observed_seconds'],1)
        self.assertIsNone(score['mean_kill_seconds'])

    def test_initially_injured_enemy_does_not_get_false_first_damage_time(self):
        t = self.tracker({'a':10})
        t.observe(snapshot({'a':0},1))
        score = summarize([t.finish()])
        self.assertEqual(score['initially_injured_zombies'],1)
        self.assertEqual(score['excluded_deaths'],1)
        self.assertIsNone(score['mean_kill_seconds'])

    def test_new_wave_ids_and_injured_first_sighting_stay_separate(self):
        t = self.tracker({'a':20})
        t.observe(snapshot({'a':0},1))
        t.observe(snapshot({'b':20,'c':5},1))
        t.observe(snapshot({'b':10,'c':0},2))
        t.observe(snapshot({'b':0},3))
        score = summarize([t.finish()])
        self.assertEqual(score['completed_zombies'],2)
        self.assertEqual(score['excluded_deaths'],1)
        self.assertEqual(score['mean_kill_seconds'],.025)

    def test_observation_gap_is_excluded_even_if_same_id_returns(self):
        t = self.tracker({'a':20})
        t.observe(snapshot({'a':10}))
        t.observe(snapshot({}))
        t.observe(snapshot({'a':5}))
        t.observe(snapshot({'a':0},1))
        score = summarize([t.finish()])
        self.assertEqual(score['observation_gap_zombies'],1)
        self.assertEqual(score['excluded_deaths'],1)
        self.assertIsNone(score['mean_kill_seconds'])

    def test_target_switches_require_an_alive_previous_target(self):
        t = self.tracker({'a':20,'b':20})
        t.observe(snapshot({'a':15,'b':20}))
        t.observe(snapshot({'a':10,'b':20}))  # same target, one opportunity
        t.observe(snapshot({'a':10,'b':15}))  # switch while a alive
        t.observe(snapshot({'a':10,'b':0},1))  # same target until b dies
        t.observe(snapshot({'a':5},1))  # choosing a after b dies is not a switch-before-kill
        score = summarize([t.finish()])
        self.assertEqual(score['counts']['focus_opportunities'],3)
        self.assertEqual(score['switch_before_kill_fraction'],1/3)
        self.assertEqual(score['same_target_fraction'],2/3)

    def test_sweep_is_ambiguous_not_sequential_target_switches(self):
        t = self.tracker({'a':20,'b':20})
        t.observe(snapshot({'a':15,'b':20}))
        t.observe(snapshot({'a':10,'b':15}))
        t.observe(snapshot({'a':10,'b':10}))
        score = summarize([t.finish()])
        self.assertIsNone(score['switch_before_kill_fraction'])
        self.assertEqual(score['counts']['multi_target_damage_ticks'],1)
        self.assertEqual(score['multi_target_damage_fraction'],1/3)

    def test_missing_previous_target_cannot_create_a_switch(self):
        t = self.tracker({'a':20,'b':20})
        t.observe(snapshot({'a':15,'b':20}))
        t.observe(snapshot({'b':10}))
        self.assertIsNone(summarize([t.finish()])['switch_before_kill_fraction'])

    def test_no_multi_enemy_opportunity_and_no_damage_are_not_zero_scores(self):
        t = self.tracker({'a':20})
        t.observe(snapshot({'a':15}))
        t.observe(snapshot({'a':10}))
        self.assertIsNone(summarize([t.finish()])['switch_before_kill_fraction'])
        t = self.tracker({'a':20})
        t.observe(snapshot({'a':20}))
        self.assertEqual(summarize([t.finish()])['damaged_zombies'],0)

    def test_pooled_mean_weights_zombies_and_reports_incomplete_cases(self):
        episodes = []
        for count,delay in [(1,20),(3,60)]:
            t = self.tracker({str(i):20 for i in range(count+1)})
            t.observe(snapshot({str(i):10 for i in range(count+1)}))
            for _ in range(delay-1):
                t.observe(snapshot({str(i):10 for i in range(count+1)}))
            t.observe(snapshot({**{str(i):0 for i in range(count)},str(count):10},count))
            episodes.append(t.finish())
        score = summarize(episodes)
        self.assertEqual(score['mean_kill_seconds'],2.5)
        self.assertEqual(score['median_kill_seconds'],3)
        self.assertEqual(score['completed_zombies'],4)
        self.assertEqual(score['alive_damaged_at_end'],2)


if __name__=='__main__':
    unittest.main()
