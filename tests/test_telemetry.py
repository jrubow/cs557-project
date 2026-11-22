"""Sound evidence must not count every frame of a lingering subtitle."""
from types import SimpleNamespace as NS
import unittest

from combat_rl.telemetry import CombatTelemetry, CRIT_KEY, TelemetryTap


def full(age=None, key=CRIT_KEY, damage=0, x=0, **kw):
    return NS(sound_subtitles=[] if age is None else [NS(translate_key=key, age=age, x=x, y=65, z=0)],
              misc_statistics={'damage_dealt':damage}, x=0, y=65, z=0,
              is_on_ground=kw.get('ground', True), velocity_y=kw.get('vy', 0))


class TelemetryTests(unittest.TestCase):
    def test_lingering_sound_is_one_event_and_refresh_is_new(self):
        tracker = CombatTelemetry()
        tracker.begin(full())
        for age in (0, 1, 2, 0, 1):
            tracker.observe(full(age), {'attack':age == 0})
        self.assertEqual(tracker.counts['critical_sound_events'], 2)
        self.assertEqual(tracker.summary()['critical_cues_per_100_attack_pulses'], 100)

    def test_baseline_sound_is_ignored_but_first_tick_age_one_is_counted(self):
        tracker = CombatTelemetry()
        tracker.begin(full(10))
        tracker.observe(full(11), {})
        tracker.observe(full(12), {})
        self.assertEqual(tracker.counts['critical_sound_events'], 0)
        tracker.observe(full(1), {})
        self.assertEqual(tracker.counts['critical_sound_events'], 1)
        tracker.begin(full())
        tracker.observe(full(1), {})
        self.assertEqual(tracker.counts['critical_sound_events'], 1)

    def test_consecutive_same_key_events_and_distant_sounds(self):
        tracker = CombatTelemetry()
        tracker.begin(full())
        for _ in range(3):
            tracker.observe(full(0, damage=60, ground=False, vy=-.2), {'attack':1, 'jump':1})
        tracker.observe(full(0, x=20), {})
        tracker.observe(full(1, x=0), {})  # Moving near a lingering cue is not a new crit.
        self.assertEqual(tracker.counts['critical_sound_events'], 3)
        self.assertEqual(tracker.counts['damage_events'], 1)
        self.assertEqual(tracker.counts['descending_attack_pulses'], 3)
        self.assertEqual(tracker.counts['airborne_ticks'], 3)

    def test_wrapper_preserves_backend_results_and_only_tracks_after_begin(self):
        class Backend:
            def reset(self, **kw):
                return {'full':full()}, {'setup':True}
            def step(self, action):
                self.action = action
                return {'full':full(0)}, 2, False, True, {'unchanged':1}
        backend = Backend()
        tap = TelemetryTap(backend)
        tap.reset()
        tap.step({'attack':1})
        tracker = tap.begin()
        result = tap.step({'attack':1})
        self.assertEqual(result[1:], (2,False,True,{'unchanged':1}))
        self.assertEqual(backend.action, {'attack':1})
        self.assertEqual(tracker.counts['critical_sound_events'], 1)
        tap.reset()
        self.assertIsNone(tap.tracker)


if __name__ == '__main__':
    unittest.main()
