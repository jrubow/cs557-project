"""Passive per-tick critical-sound diagnostics; never part of policy input."""
import math

CRIT_KEY = 'subtitles.entity.player.attack.crit'


class CombatTelemetry:
    """Count sound age resets, not every frame of a lingering subtitle.

    The pinned CraftGround listener retains one entry per subtitle key, ages it
    each tick and resets age on a new sound. Multiple identical sounds in one
    tick coalesce. These are sound-cue counts, not an instrumented server hit
    counter. Validate with positive/negative controls before interpreting zeros.
    """
    def begin(self, full):
        self.tick = 0
        self.births = {s.translate_key: -int(s.age) for s in full.sound_subtitles}
        self.damage = float(full.misc_statistics['damage_dealt'])/10
        self.counts = dict(critical_sound_events=0, attack_pulses=0, damage_events=0,
                           jump_ticks=0, airborne_ticks=0, descending_attack_pulses=0)
        self.sound_counts = {}
        self.events = []

    def observe(self, full, action):
        self.tick += 1
        pulses = int(bool(action.get('attack')))
        airborne = not full.is_on_ground
        descending = airborne and full.velocity_y < -1e-5
        dealt = float(full.misc_statistics['damage_dealt'])/10
        delta = max(0, dealt-self.damage)
        self.damage = dealt
        sounds = []
        for sound in full.sound_subtitles:
            key = sound.translate_key
            birth = self.tick-int(sound.age)
            previous = self.births.get(key)
            self.births[key] = max(birth, previous) if previous is not None else birth
            # The listener may age a just-played sound to 1 before observation.
            if birth < 0 or (previous is not None and birth <= previous):
                continue
            distance = math.sqrt((sound.x-full.x)**2+(sound.y-full.y)**2+(sound.z-full.z)**2)
            if key.startswith('subtitles.entity.player.attack.') and distance <= 3:
                sounds.append(key)
                self.sound_counts[key] = self.sound_counts.get(key, 0)+1
        crit = int(CRIT_KEY in sounds)
        self.counts['critical_sound_events'] += crit
        self.counts['attack_pulses'] += pulses
        self.counts['damage_events'] += int(delta > 0)
        self.counts['jump_ticks'] += int(bool(action.get('jump')))
        self.counts['airborne_ticks'] += int(airborne)
        self.counts['descending_attack_pulses'] += int(descending and pulses)
        if pulses or delta or sounds:
            self.events.append(dict(tick=self.tick, game_seconds=self.tick/20,
                critical_sound=crit, attack_pulse=pulses, jump=int(bool(action.get('jump'))),
                on_ground=bool(full.is_on_ground), velocity_y=full.velocity_y,
                damage_dealt_delta=delta, sounds=sounds))
        return crit

    def summary(self):
        return dict(self.counts, ticks=self.tick, sounds=self.sound_counts,
            critical_cues_per_100_attack_pulses=(100*self.counts['critical_sound_events']/self.counts['attack_pulses']
                                               if self.counts['attack_pulses'] else None),
            critical_cues_per_game_minute=(1200*self.counts['critical_sound_events']/self.tick if self.tick else None),
            measurement='Deduplicated nearby critical-attack sound cues; not a server-side hit counter.')


class TelemetryTap:
    def __init__(self, backend):
        self.backend, self.latest, self.tracker = backend, None, None

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def reset(self, **kwargs):
        self.tracker = None
        result = self.backend.reset(**kwargs)
        self.latest = result[0]
        return result

    def step(self, action):
        result = self.backend.step(action)
        self.latest = result[0]
        if self.tracker is not None:
            self.tracker.observe(result[0]['full'], action)
        return result

    def begin(self):
        self.tracker = CombatTelemetry()
        self.tracker.begin(self.latest['full'])
        return self.tracker
