"""Test behavioral evidence against confounders, geometry, and censoring."""
import math
from types import SimpleNamespace as NS
import unittest

from combat_rl.behavior import BehaviorTracker,aggregate,geometry,visible_targets
from combat_rl.config import load_config
from combat_rl.env import ENTITY_RANGE,read_state


def full(x=0,z=0,yaw=0,pitch=0,enemies=None,dealt=0,taken=0,health=20):
    enemies = [(0,2.5)] if enemies is None else enemies
    entities = [NS(unique_name=str(i),translation_key='entity.minecraft.zombie',
                   x=ex,y=65,z=ez,health=20) for i,(ex,ez) in enumerate(enemies)]
    return NS(x=x,y=65,z=z,yaw=yaw,pitch=pitch,health=health,is_dead=health<=0,
        misc_statistics={'damage_dealt':dealt,'damage_taken':taken},killed_statistics={'zombie':0},
        surrounding_entities={ENTITY_RANGE:NS(entities=entities)},sound_subtitles=[],
        is_on_ground=True,velocity_y=0)


class BehaviorTests(unittest.TestCase):
    def tracker(self,initial):
        tracker = BehaviorTracker(load_config())
        tracker.begin(initial)
        return tracker

    def test_yaw_correction_measures_camera_not_target_motion(self):
        tracker = self.tracker(full(enemies=[(-3,3)]))
        tracker.observe(full(yaw=30,enemies=[(-3,3)]),{'camera_yaw':30})
        score = aggregate([tracker.finish()])
        self.assertEqual(score['yaw_correction_fraction'],1)
        self.assertAlmostEqual(score['mean_camera_only_yaw_gain_degrees'],30)
        tracker = self.tracker(full(enemies=[(-3,3)]))
        row = tracker.observe(full(enemies=[(0,3)]),{})
        self.assertEqual(row['target_error_after'],0)
        self.assertEqual(row['yaw_rotation_gain'],0)
        self.assertIsNone(aggregate([tracker.finish()])['yaw_correction_fraction'])

    def test_wraparound_and_hidden_enemy_filter(self):
        angle = math.radians(-161)
        enemy = (-3*math.sin(angle),3*math.cos(angle))
        tracker = self.tracker(full(yaw=179,enemies=[enemy]))
        row = tracker.observe(full(yaw=-179,enemies=[enemy]),{'camera_yaw':2})
        self.assertAlmostEqual(row['yaw_rotation_gain'],2)
        state = read_state({'full':full(enemies=[(0,-1),(0,3)])})
        targets = visible_targets(state,load_config()['environment'],'C')
        self.assertEqual([e.uuid for e in targets],['1'])

    def test_pitch_correction_and_attack_band_are_geometric(self):
        tracker = self.tracker(full(pitch=-20))
        row = tracker.observe(full(pitch=-10),{'camera_pitch':10,'attack':1})
        self.assertAlmostEqual(row['pitch_rotation_gain'],10)
        summary = aggregate([tracker.finish()])
        self.assertEqual(summary['pitch_correction_fraction'],1)
        self.assertEqual(summary['aligned_attack_fraction'],0)  # Pre-action aim was above the target.

    def test_kiting_requires_agent_motion_and_an_unanswered_damage_window(self):
        for motion,hurt,expected in [(True,False,1),(False,False,0),(True,True,0)]:
            tracker = self.tracker(full())
            for tick in range(1,21):
                z = -.05*tick if motion else 0
                # Either both pursue/retreat at the same speed, or only the enemy moves away.
                enemy_z = 2.5+z if motion else 2.5+.04*tick
                tracker.observe(full(z=z,enemies=[(0,enemy_z)],dealt=70 if tick>=10 else 0,
                    taken=10 if hurt and tick==20 else 0),{'back':1,'attack':int(tick in (1,13))})
            summary = aggregate([tracker.finish()])
            self.assertEqual(summary['movement_windows']['retreat']['unanswered_damage_windows'],expected)
            if not motion:
                self.assertEqual(summary['movement_windows']['retreat']['windows'],0)
            self.assertEqual(summary['damage_per_attack_pulse'],3.5)

    def test_partial_windows_and_unfinished_searches_are_not_fake_successes(self):
        tracker = self.tracker(full(enemies=[(0,-2.5)]))
        for _ in range(5):
            tracker.observe(full(enemies=[(0,-2.5)]),{'back':1,'attack':1})
        episode = tracker.finish()
        score = aggregate([episode])
        self.assertEqual(score['partial_windows'],1)
        self.assertEqual(score['movement_windows']['retreat']['windows'],0)
        self.assertEqual(score['search_censored'],1)
        self.assertIsNone(score['median_reacquisition_seconds'])
        self.assertEqual(episode,tracker.finish())  # Finalization is idempotent.
        tracker = self.tracker(full(enemies=[(0,-2.5)]))
        tracker.observe(full(yaw=180,enemies=[(0,-2.5)]),{'camera_yaw':180})
        score = aggregate([tracker.finish()])
        self.assertEqual(score['search_acquired'],1)
        self.assertEqual(score['median_reacquisition_seconds'],.05)

    def test_wall_escape_uses_hysteresis_and_death_is_censored(self):
        tracker = self.tracker(full(x=10.5))
        tracker.observe(full(x=9.5),{})
        self.assertEqual(tracker.walls,[])
        tracker.observe(full(x=8.9),{})
        self.assertEqual(aggregate([tracker.finish()])['wall_escapes'],1)
        tracker = self.tracker(full(x=10.5))
        tracker.observe(full(x=10.5,health=0,taken=200),{})
        summary = aggregate([tracker.finish()])
        self.assertEqual(summary['wall_censored'],1)
        self.assertEqual(summary['wall_escapes'],0)

    def test_disappearing_targets_do_not_look_like_reacquisition_or_alignment(self):
        tracker = self.tracker(full(enemies=[(0,-2)]))
        tracker.observe(full(enemies=[]),{'attack':1})
        summary = aggregate([tracker.finish()])
        self.assertEqual(summary['search_acquired'],0)
        self.assertEqual(summary['aligned_attack_fraction'],0)
        self.assertEqual(summary['attacks_without_visible_target_fraction'],1)

    def test_focus_tracks_hidden_uuid_health_and_raw_corpses(self):
        tracker = self.tracker(full(enemies=[(0,-2)]))
        damaged = full(enemies=[(0,-2)],dealt=100)
        damaged.surrounding_entities[ENTITY_RANGE].entities[0].health = 10
        tracker.observe(damaged,{})
        dead = full(enemies=[(0,-2)],dealt=200)
        dead.surrounding_entities[ENTITY_RANGE].entities[0].health = 0
        dead.killed_statistics['zombie'] = 1
        row = tracker.observe(dead,{})
        self.assertEqual(read_state({'full':dead}).entities,[])
        self.assertEqual(row['death_ids'],'["0"]')
        score = aggregate([tracker.finish()])
        self.assertEqual(score['focus_fire']['mean_kill_seconds'],.05)
        self.assertEqual(score['focus_fire']['counts']['player_kills'],1)
        self.assertEqual(score['counts']['visible_ticks'],0)


if __name__=='__main__':
    unittest.main()
