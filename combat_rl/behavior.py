"""Passive combat behavior measurements, with explicit geometric proxies."""
from collections import Counter
import json
import math
import statistics

from .env import read_state
from .focus import FocusTracker, DEFINITIONS as FOCUS_DEFINITIONS, summarize as summarize_focus

DEFINITIONS = dict(
    focus_fire=FOCUS_DEFINITIONS,
    close_distance=4.0, aligned_yaw_degrees=15.0, aligned_pitch_degrees=20.0,
    player_eye_offset=1.62, zombie_torso_offset=1.0, radial_motion_per_tick=.01,
    window_ticks=20, near_wall_gap=2.0, escaped_wall_gap=3.0,
    note='Angles target the nearest policy-visible zombie torso, not an exact crosshair/hitbox raycast. '
         'Range 4 is a diagnostic band, not a claim about attack reach. '
         'Damage is player-attributed statistic change; timing may lag attacks. '
         'Unanswered retreat windows are evidence consistent with kiting, not proof of intent or invulnerability.')


def wrap(degrees):
    return (degrees+180) % 360-180


def geometry(state, enemy):
    dx, dz = enemy.x-state.x, enemy.z-state.z
    distance = math.hypot(dx, dz)
    bearing = wrap(math.degrees(math.atan2(-dx, dz))-state.yaw)
    pitch_error = math.degrees(math.atan2(state.y+1.62-(enemy.y+1), max(distance,.001)))-state.pitch
    return distance, bearing, pitch_error


def visible_targets(state, settings, condition):
    targets = [e for e in state.entities if condition != 'C'
               or abs(geometry(state,e)[1]) <= settings['fov_degrees']/2]
    return sorted(targets, key=lambda e: ((e.x-state.x)**2+(e.y-state.y)**2+(e.z-state.z)**2,e.uuid))[:5]


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def aggregate(episodes):
    """Pool opportunities, retain episode summaries for uncertainty/context checks."""
    counts = Counter()
    windows, searches, walls, intervals = [], [], [], []
    for episode in episodes:
        counts.update(episode['counts'])
        windows.extend(episode['windows'])
        searches.extend(episode['searches'])
        walls.extend(episode['walls'])
        intervals.extend(episode['attack_intervals_ticks'])
    groups = {}
    for group in ('retreat','approach','strafe','other'):
        rows = [w for w in windows if not w['partial'] and w['group']==group]
        groups[group] = dict(windows=len(rows), seconds=len(rows),
            damage_dealt=sum(w['damage_dealt'] for w in rows), damage_taken=sum(w['damage_taken'] for w in rows),
            unanswered_damage_windows=sum(w['damage_dealt']>0 and w['damage_taken']==0 for w in rows),
            mean_enemy_count=statistics.mean(w['mean_enemy_count'] for w in rows) if rows else None,
            mean_start_health=statistics.mean(w['start_health'] for w in rows) if rows else None)
    completed_searches = [e for e in searches if e['outcome']=='acquired']
    escaped = [e for e in walls if e['outcome']=='escaped']
    c = counts
    return dict(episodes=len(episodes), counts=dict(counts), movement_windows=groups,
        focus_fire=summarize_focus([e['focus_fire'] for e in episodes]),
        yaw_correction_fraction=ratio(c['yaw_improving_turns'],c['yaw_turns']),
        pitch_correction_fraction=ratio(c['pitch_improving_turns'],c['pitch_turns']),
        mean_camera_only_yaw_gain_degrees=ratio(c['yaw_gain_sum'],c['yaw_turns']),
        mean_abs_visible_yaw_error=ratio(c['yaw_error_sum'],c['visible_ticks']),
        mean_abs_visible_pitch_error=ratio(c['pitch_error_sum'],c['visible_ticks']),
        aligned_attack_fraction=ratio(c['aligned_attacks'],c['attack_pulses']),
        attacks_without_visible_target_fraction=ratio(c['attacks_without_visible_target'],c['attack_pulses']),
        damage_per_attack_pulse=ratio(c['damage_dealt'],c['attack_pulses']),
        critical_cues_per_100_attack_pulses=ratio(100*c['critical_cues'],c['attack_pulses']),
        median_attack_interval_ticks=statistics.median(intervals) if intervals else None,
        airborne_attack_fraction=ratio(c['airborne_attacks'],c['attack_pulses']),
        near_wall_fraction=ratio(c['near_wall_ticks'],c['ticks']),
        near_wall_damage_per_second=ratio(20*c['near_wall_damage_taken'],c['near_wall_ticks']),
        away_wall_damage_per_second=ratio(20*(c['damage_taken']-c['near_wall_damage_taken']),c['ticks']-c['near_wall_ticks']),
        search_intervals=len(searches), search_acquired=len(completed_searches),
        search_censored=sum(e['outcome']=='episode_end' for e in searches),
        median_reacquisition_seconds=statistics.median(e['seconds'] for e in completed_searches) if completed_searches else None,
        wall_intervals=len(walls), wall_escapes=len(escaped),
        wall_escapes_within_5_seconds=sum(e['seconds']<=5 for e in escaped),
        wall_censored=sum(e['outcome']=='episode_end' for e in walls),
        partial_windows=sum(w['partial'] for w in windows))


class BehaviorTracker:
    def __init__(self, config, condition='C', emit=None):
        self.settings, self.condition, self.emit = config['environment'], condition, emit

    def gap(self, state):
        s = self.settings
        return min(s['arena_radius']-abs(state.x-s['arena_center_x']),
                   s['arena_radius']-abs(state.z-s['arena_center_z']))

    def begin(self, full):
        self.previous = read_state({'full':full})
        self.focus = FocusTracker()
        self.focus.begin(full)
        self.tick, self.counts, self.last_row = 0, Counter(), None
        self.windows, self.searches, self.walls, self.attack_intervals = [], [], [], []
        self.buffer = []
        self.last_attack = None
        self.search_start = 0 if self.previous.entities and not visible_targets(self.previous,self.settings,self.condition) else None
        self.search_turn = 0.0
        self.wall_start = 0 if self.gap(self.previous)<=2 else None
        self.finished = False

    def observe(self, full, action, critical_cue=0):
        if self.finished:
            raise RuntimeError('Begin a new episode before observing more ticks')
        before, after = self.previous, read_state({'full':full})
        self.tick += 1
        focus = self.focus.observe(full)
        targets = visible_targets(before,self.settings,self.condition)
        target = targets[0] if targets else None
        pulse = int(bool(action.get('attack')))
        dealt, taken = max(0,after.damage_dealt-before.damage_dealt), max(0,after.damage_taken-before.damage_taken)
        yaw_change = wrap(after.yaw-before.yaw)
        row = dict(tick=self.tick, seconds=self.tick/20, x=before.x,y=before.y,z=before.z,
            x_after=after.x,y_after=after.y,z_after=after.z,yaw=before.yaw,pitch=before.pitch,
            yaw_after=after.yaw,pitch_after=after.pitch,health=before.health,health_after=after.health,
            attack=pulse,back=int(bool(action.get('back'))),forward=int(bool(action.get('forward'))),
            strafe=int(bool(action.get('left') or action.get('right'))),jump=int(bool(action.get('jump'))),
            sprint=int(bool(action.get('sprint'))),camera_yaw=action.get('camera_yaw',0),
            camera_pitch=action.get('camera_pitch',0),yaw_change=yaw_change,
            on_ground_after=bool(full.is_on_ground),velocity_y_after=float(full.velocity_y),
            damage_dealt=dealt,damage_taken=taken,kills=max(0,after.kills-before.kills),critical_cue=critical_cue,
            enemy_count=len(before.entities),visible_count=len(targets),
            damaged_ids=json.dumps(focus['damaged_ids']),death_ids=json.dumps(focus['death_ids']),
            wounded_alive=focus['wounded_alive'],damage_target_transition=focus['transition'],
            entity_health_after=json.dumps(self.focus.previous,separators=(',',':')),
            nearest_any_distance=min((geometry(before,e)[0] for e in before.entities),default=None),
            target_id=target.uuid if target else None,target_distance=None,target_bearing=None,target_pitch_error=None,
            target_alive_after=False,target_error_after=None,target_pitch_error_after=None,
            yaw_rotation_gain=None,pitch_rotation_gain=None,
            radial_away=0.0,tangential_motion=0.0,close=False,aligned=False,wall_gap=self.gap(before),
            entities=json.dumps([dict(id=e.uuid,x=e.x,y=e.y,z=e.z,health=e.health) for e in before.entities],separators=(',',':')))
        c = self.counts
        c.update(ticks=1,enemies_ticks=int(bool(before.entities)),visible_ticks=int(bool(targets)),
                 attack_pulses=pulse,damage_dealt=dealt,damage_taken=taken,critical_cues=critical_cue,
                 back_ticks=row['back'],strafe_ticks=row['strafe'],
                 airborne_attacks=pulse*int(not full.is_on_ground))
        if target is not None:
            distance,bearing,pitch = geometry(before,target)
            dx,dz = target.x-before.x,target.z-before.z
            mx,mz = after.x-before.x,after.z-before.z
            radial = -(mx*dx+mz*dz)/max(distance,.001)
            tangent = abs(mx*dz-mz*dx)/max(distance,.001)
            gain = abs(bearing)-abs(wrap(bearing-yaw_change))
            pitch_change = after.pitch-before.pitch
            pitch_gain = abs(pitch)-abs(pitch-pitch_change)
            match = next((e for e in after.entities if e.uuid==target.uuid),None)
            row.update(target_distance=distance,target_bearing=bearing,target_pitch_error=pitch,
                target_alive_after=match is not None,target_error_after=geometry(after,match)[1] if match else None,
                target_pitch_error_after=geometry(after,match)[2] if match else None,
                yaw_rotation_gain=gain,pitch_rotation_gain=pitch_gain,radial_away=radial,tangential_motion=tangent,
                close=distance<=4 and abs(bearing)<=45,
                aligned=distance<=4 and abs(bearing)<=15 and abs(pitch)<=20)
            c.update(yaw_error_sum=abs(bearing),pitch_error_sum=abs(pitch),
                aligned_attacks=pulse*int(row['aligned']),close_ticks=int(row['close']),
                radial_retreat_ticks=int(radial>.01),tangential_ticks=int(tangent>.01),
                attacks_outside_close=pulse*int(not row['close']))
            if abs(yaw_change)>.01:
                c.update(yaw_turns=1,yaw_improving_turns=int(gain>1),yaw_worsening_turns=int(gain < -1),yaw_gain_sum=gain)
            if abs(pitch_change)>.01:
                c.update(pitch_turns=1,pitch_improving_turns=int(pitch_gain>1),pitch_gain_sum=pitch_gain)
        else:
            c['attacks_without_visible_target'] += pulse
        if pulse:
            if self.last_attack is not None:
                self.attack_intervals.append(self.tick-self.last_attack)
            self.last_attack = self.tick
        if row['wall_gap']<=2:
            c.update(near_wall_ticks=1,near_wall_damage_taken=taken)
        current_visible = visible_targets(after,self.settings,self.condition)
        if self.search_start is not None:
            self.search_turn += abs(yaw_change)
            if current_visible or not after.entities:
                self._end_search('acquired' if current_visible else 'enemies_gone')
        elif after.entities and not current_visible:
            self.search_start, self.search_turn = self.tick, 0.0
        if self.wall_start is not None and self.gap(after)>=3:
            self._end_wall('escaped')
        elif self.wall_start is None and self.gap(after)<=2:
            self.wall_start = self.tick
        self.buffer.append(row)
        if len(self.buffer)==20:
            self._window()
        self.last_row, self.previous = row, after
        if self.emit is not None:
            self.emit(row)
        return row

    def _end_search(self, outcome):
        self.searches.append(dict(start_tick=self.search_start,end_tick=self.tick,
            seconds=(self.tick-self.search_start)/20,outcome=outcome,camera_yaw_travel=self.search_turn))
        self.search_start = None

    def _end_wall(self, outcome):
        self.walls.append(dict(start_tick=self.wall_start,end_tick=self.tick,
            seconds=(self.tick-self.wall_start)/20,outcome=outcome))
        self.wall_start = None

    def _window(self):
        rows = self.buffer
        n = len(rows)
        close = sum(r['close'] for r in rows)
        back = sum(r['back'] for r in rows)
        forward = sum(r['forward'] for r in rows)
        strafe = sum(r['strafe'] for r in rows)
        away = sum(r['radial_away']>.01 for r in rows)
        toward = sum(r['radial_away']<-.01 for r in rows)
        tangent = sum(r['tangential_motion']>.01 for r in rows)
        attacks = sum(r['attack'] for r in rows)
        group = 'other'
        if close>=10 and attacks:
            if back>=10 and away>=6:
                group = 'retreat'
            elif forward>=10 and toward>=6:
                group = 'approach'
            elif strafe>=10 and tangent>=6:
                group = 'strafe'
        self.windows.append(dict(start_tick=rows[0]['tick']-1,end_tick=rows[-1]['tick'],partial=n!=20,
            group=group,damage_dealt=sum(r['damage_dealt'] for r in rows),damage_taken=sum(r['damage_taken'] for r in rows),
            attack_pulses=attacks,close_ticks=close,back_ticks=back,radial_away_ticks=away,
            mean_enemy_count=statistics.mean(r['enemy_count'] for r in rows),start_health=rows[0]['health']))
        self.buffer = []

    def finish(self):
        if not self.finished:
            if self.buffer:
                self._window()
            if self.search_start is not None:
                self._end_search('episode_end')
            if self.wall_start is not None:
                self._end_wall('episode_end')
            self.finished = True
        return dict(counts=dict(self.counts),windows=self.windows,searches=self.searches,walls=self.walls,
                    attack_intervals_ticks=self.attack_intervals,focus_fire=self.focus.finish())
