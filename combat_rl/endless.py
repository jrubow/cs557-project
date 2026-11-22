"""Continuous or cleared-batch survival tasks, separate from five-wave scores."""
import math

from .env import CombatEnv, read_state
from .task import NO_OP_ACTION, action_to_craftground, reward


class EndlessCombatEnv(CombatEnv):
    """Timed arrivals or fixed-size batches replenished only after a full clear.

    Keep the trained observation/action spaces. The wave feature is fixed at 5
    for timed arrivals, or k for batches; the nearest-five/FOV filter applies.
    Extra enemies never expand the policy's input or become privileged features.
    """
    def __init__(self, config, condition="C", *, spawn_seconds=2.0, max_seconds=600,
                 port=8000, backend=None, mode='timed', batch_size=1):
        if mode not in ('timed', 'batch') or type(batch_size) is not int or not 1 <= batch_size <= 5:
            raise ValueError('Use timed/batch mode and integer batch_size in [1,5]')
        for name, value in (("spawn_seconds", spawn_seconds), ("max_seconds", max_seconds)):
            if not math.isfinite(value) or value <= 0 or abs(value*20-round(value*20)) > 1e-8:
                raise ValueError(f"{name} must be positive and a whole number of game ticks")
        if max_seconds > 600:
            raise ValueError("Demonstration recording limit is at most ten game minutes")
        super().__init__(config, condition=condition, evaluation=True, port=port, backend=backend)
        self.spawn_ticks = round(spawn_seconds*20)
        self.max_ticks = round(max_seconds*20)
        self.spawn_commands_issued = 0
        self.peak_alive = 0
        self.mode, self.batch_size = mode, batch_size
        self.batches_completed = 0

    def _initial_enemy_count(self):
        return self.batch_size if self.mode == 'batch' else 1

    def reset(self, *, seed=None, options=None):
        self.spawn_commands_issued = self.peak_alive = self._initial_enemy_count()
        self.batches_completed = 0
        super().reset(seed=seed, options=options)
        self._batch_kills_start = self.state.kills
        self.wave, self.wave_target = (self.batch_size if self.mode == 'batch' else 5), 0
        self.history.clear()
        for _ in range(self.history.maxlen):
            observation = self._observation(NO_OP_ACTION)
        return observation, self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a finished episode")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid compact combat action: {action}")
        command = action_to_craftground(action)
        total_reward, terminated = 0.0, False
        for tick in range(self.settings["action_repeat"]):
            # Queue the summon for the tick at the exact interval boundary.
            if self.mode == 'timed' and (self.elapsed_ticks+1) % self.spawn_ticks == 0:
                self._raw.add_commands(self._spawn_commands(1))
                self.spawn_commands_issued += 1
            tick_action = dict(command)
            if tick:
                tick_action["camera_yaw"] = tick_action["camera_pitch"] = 0.0
            tick_action["attack"] = int(tick == 0 and command["attack"] and not self._attack_held)
            self._attack_held = bool(tick_action["attack"])
            raw, *_ = self._raw.step(tick_action)
            previous, self.state = self.state, read_state(raw)
            self.elapsed_ticks += 1
            self.peak_alive = max(self.peak_alive, len(self.state.entities))
            dead = self.state.is_dead or self.state.health <= 0
            cleared = (self.mode == 'batch' and not dead and not self.state.entities
                       and self.state.kills-self._batch_kills_start >= self.batch_size)
            value, deltas = reward(previous, self.state, cleared, dead, self.config,
                                   shaped=self.condition != "A", elapsed_ticks=1)
            total_reward += value
            for key in self.totals:
                self.totals[key] += deltas[key]
            if dead:
                terminated, self.terminal_reason = True, "death"
            elif cleared:
                self.batches_completed += 1
                self._batch_kills_start = self.state.kills
                if self.elapsed_ticks < self.max_ticks:
                    self._raw.add_commands(self._spawn_commands(self.batch_size))
                    self.spawn_commands_issued += self.batch_size
            if terminated or self.elapsed_ticks >= self.max_ticks:
                break
        self.elapsed_steps += 1
        truncated = not terminated and self.elapsed_ticks >= self.max_ticks
        if truncated:
            self.terminal_reason = "recording_time_limit"
        self._done = terminated or truncated
        return self._observation(action), total_reward, terminated, truncated, self._info()

    def _info(self):
        info = super()._info()
        info.update(task='endless_batch' if self.mode == 'batch' else 'endless_demo',
                    mode=self.mode, batch_size=self.batch_size, batches_completed=self.batches_completed,
                    success=False, waves_completed=0, wave_target=0,
                    observation_wave=self.wave, spawn_interval_seconds=self.spawn_ticks/20,
                    recording_limit_seconds=self.max_ticks/20,
                    spawn_commands_issued=self.spawn_commands_issued,
                    active_enemies=len(self.state.entities) if self.state is not None else 0,
                    peak_active_enemies=self.peak_alive,
                    final_health=self.state.health if self.state is not None else None,
                    kills_per_minute=self.totals["kills"]*1200/self.elapsed_ticks if self.elapsed_ticks else 0)
        return info
