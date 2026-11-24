"""Passive UUID/health observations of focus fire; never infer death from absence."""
from collections import Counter
import statistics

from .env import ENTITY_RANGE


DEFINITIONS = dict(
    full_zombie_health=20.0, game_ticks_per_second=20,
    clock='First observed health decrease to first observed health <= 0, by UUID.',
    inclusion='Mean/median require a full-health observation before first damage and no observation gaps.',
    censoring='Alive at episode end and disappearance are incomplete, not kills. Initially injured '
              'or observation-gap records remain separate. One-tick lethal damage has duration zero.',
    focus='Consecutive single-target health-loss events, with the previous damaged target still '
          'alive and at least two live zombies before the next event. Multi-target events break the chain.',
    note='Client-observed health changes at 0.05-second resolution, not server hit timestamps or '
         'player-attributed per-target attacks. Sweeps/environmental damage can affect health. '
         'Completed-only kill time is subject to survivor selection; inspect incomplete cases and '
         'damage-target switches too. All arena entities are diagnostic only, including outside policy FOV.')


def healths(full):
    return {e.unique_name: float(e.health)
            for e in full.surrounding_entities[ENTITY_RANGE].entities
            if e.translation_key == 'entity.minecraft.zombie'}


def summarize(episodes):
    records, counts = [], Counter()
    for episode in episodes:
        records.extend(episode['zombies'])
        counts.update(episode['counts'])
    completed = [r for r in records if r['kill_seconds'] is not None]
    durations = [r['kill_seconds'] for r in completed]
    damaged = [r for r in records if r['first_damage_tick'] is not None or r['initially_injured']]
    incomplete = [r for r in damaged if r['death_tick'] is None]
    divide = lambda n,d: n/d if d else None
    return dict(
        counts=dict(counts), tracked_zombies=len(records), damaged_zombies=len(damaged),
        completed_zombies=len(completed),
        mean_kill_seconds=statistics.mean(durations) if durations else None,
        median_kill_seconds=statistics.median(durations) if durations else None,
        incomplete_damaged_zombies=len(incomplete),
        alive_damaged_at_end=sum(r['outcome']=='episode_end' for r in incomplete),
        lost_damaged_zombies=sum(r['outcome']=='observation_lost' for r in incomplete),
        initially_injured_zombies=sum(r['initially_injured'] for r in records),
        observation_gap_zombies=sum(r['observation_gap'] for r in records),
        health_confirmed_deaths=sum(r['death_tick'] is not None for r in records),
        excluded_deaths=sum(r['death_tick'] is not None and r['kill_seconds'] is None for r in records),
        switch_before_kill_fraction=divide(counts['switches_before_kill'],counts['focus_opportunities']),
        same_target_fraction=divide(counts['same_target_events'],counts['focus_opportunities']),
        multi_target_damage_fraction=divide(counts['multi_target_damage_ticks'],counts['damage_ticks']),
        mean_wounded_alive=divide(counts['wounded_alive_sum'],counts['ticks']),
        multiple_wounded_fraction=divide(counts['multiple_wounded_ticks'],counts['multi_enemy_ticks']))


class FocusTracker:
    def begin(self, full):
        self.tick, self.counts, self.records, self.events = 0, Counter(), {}, []
        self.previous, self.last_target = healths(full), None
        self.previous_kills = int(full.killed_statistics['zombie'])
        self.finished = False
        for uid, hp in self.previous.items():
            if hp > 0:
                self._new(uid, hp)

    def _new(self, uid, hp):
        self.records[uid] = dict(id=uid, first_seen_tick=self.tick, initial_health=hp,
            initially_injured=hp < 20-1e-5, observation_gap=False, first_damage_tick=None,
            last_seen_tick=self.tick, last_health=hp, damage_events=0, health_loss=0.0,
            death_tick=None, kill_seconds=None, observed_seconds=None, outcome='tracking')

    def observe(self, full):
        if self.finished:
            raise RuntimeError('Begin a new focus episode before observing')
        self.tick += 1
        current, hits, deaths = healths(full), [], []
        for uid, hp in current.items():
            if uid not in self.records:
                if hp <= 0:  # A previously unobserved corpse has no measurable start.
                    continue
                self._new(uid, hp)
            record = self.records[uid]
            if record['death_tick'] is not None:
                continue  # A lingering corpse is one death, not one per tick.
            if uid not in self.previous and record['first_seen_tick'] != self.tick:
                record['observation_gap'] = True
            previous_hp = self.previous.get(uid)
            if previous_hp is not None and previous_hp-hp > 1e-5:
                if record['first_damage_tick'] is None:
                    record['first_damage_tick'] = self.tick
                record['damage_events'] += 1
                record['health_loss'] += previous_hp-max(0,hp)
                hits.append(uid)
            record.update(last_seen_tick=self.tick, last_health=hp, outcome='tracking')
            if hp <= 0:
                record.update(death_tick=self.tick, outcome='health_zero')
                start = record['first_damage_tick']
                if start is not None and not record['initially_injured'] and not record['observation_gap']:
                    record['kill_seconds'] = (self.tick-start)/20
                deaths.append(uid)
        for uid, record in self.records.items():
            if uid not in current and record['death_tick'] is None:
                record.update(observation_gap=True, outcome='observation_lost')

        transition = None
        if self.last_target not in self.previous or self.previous.get(self.last_target,0) <= 0:
            self.last_target = None
        if hits:
            self.counts['damage_ticks'] += 1
            self.counts['entity_health_loss_events'] += len(hits)
            if len(hits) > 1:
                self.counts['multi_target_damage_ticks'] += 1
                self.last_target = None  # No ordering/primary target is observable for a sweep.
            else:
                uid = hits[0]
                previous_still_known = (uid == self.last_target or current.get(self.last_target,0)>0)
                if (self.last_target is not None and previous_still_known
                        and sum(hp>0 for hp in self.previous.values()) >= 2):
                    transition = 'same_target' if uid == self.last_target else 'switch_before_kill'
                    self.counts['focus_opportunities'] += 1
                    self.counts['same_target_events' if uid==self.last_target else 'switches_before_kill'] += 1
                self.last_target = uid if current[uid]>0 else None
        wounded = sum(0 < hp < 20-1e-5 for hp in current.values())
        live = sum(hp>0 for hp in current.values())
        kills = int(full.killed_statistics['zombie'])
        self.counts.update(ticks=1, wounded_alive_sum=wounded, multi_enemy_ticks=int(live>=2),
            multiple_wounded_ticks=int(wounded>=2), player_kills=max(0,kills-self.previous_kills))
        self.previous_kills = kills
        row = dict(damaged_ids=sorted(hits), death_ids=sorted(deaths), wounded_alive=wounded,
                   transition=transition)
        if hits or deaths:
            self.events.append(dict(tick=self.tick, **row))
        self.previous = current
        return row

    def finish(self):
        if not self.finished:
            for record in self.records.values():
                if record['outcome']=='tracking':
                    record['outcome'] = 'episode_end'
                start = record['first_damage_tick']
                if start is not None:
                    endpoint = record['death_tick'] if record['death_tick'] is not None else record['last_seen_tick']
                    record['observed_seconds'] = (endpoint-start)/20
            self.finished = True
        return dict(counts=dict(self.counts), zombies=list(self.records.values()), events=self.events)
