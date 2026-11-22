"""Evaluation-only continuous-spawn demonstration; separate from five-wave scores."""
import math

from .env import CombatEnv, read_state
from .task import NO_OP_ACTION, action_to_craftground, reward


class EndlessCombatEnv(CombatEnv):
    """One initial zombie, then one per fixed interval, with no wave-clear reset.

    Keep the trained observation/action spaces. The wave feature is fixed at 5
    (the final trained stage); the usual nearest-five/FOV filter still applies.
    Extra enemies never expand the policy's input or become privileged features.
    """
    def __init__(self, config, condition="C", *, spawn_seconds=2.0, max_seconds=600,
                 port=8000, backend=None):
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

    def reset(self, *, seed=None, options=None):
        self.spawn_commands_issued, self.peak_alive = 1, 1
        super().reset(seed=seed, options=options)
        self.wave, self.wave_target = 5, 0
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
            if (self.elapsed_ticks+1) % self.spawn_ticks == 0:
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
            value, deltas = reward(previous, self.state, False, dead, self.config,
                                   shaped=self.condition != "A", elapsed_ticks=1)
            total_reward += value
            for key in self.totals:
                self.totals[key] += deltas[key]
            if dead:
                terminated, self.terminal_reason = True, "death"
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
        info.update(task="endless_demo", success=False, waves_completed=0, wave_target=0,
                    observation_wave=5, spawn_interval_seconds=self.spawn_ticks/20,
                    recording_limit_seconds=self.max_ticks/20,
                    spawn_commands_issued=self.spawn_commands_issued,
                    active_enemies=len(self.state.entities) if self.state is not None else 0,
                    peak_active_enemies=self.peak_alive,
                    final_health=self.state.health if self.state is not None else None,
                    kills_per_minute=self.totals["kills"]*1200/self.elapsed_ticks if self.elapsed_ticks else 0)
        return info
