"""A small Gymnasium task wrapper around the real CraftGround simulator."""

from collections import deque
import math

import gymnasium as gym
import numpy as np

from .task import (COMPACT_NVECS, FRAME_SIZE, NO_OP_ACTION, Entity, State,
                   action_to_craftground, encode_frame, normalize_action, reward)

ENTITY_RANGE = 64  # Covers opposite corners, even at the largest supported arena size.
FLOOR_Y = 64


def arena_commands(config):
    e = config["environment"]
    r, x, z = e["arena_radius"], e["arena_center_x"], e["arena_center_z"]
    return [
        "gamerule doMobSpawning false", "gamerule naturalRegeneration false",
        "gamerule doDaylightCycle false", "gamerule doWeatherCycle false",
        "gamerule doMobLoot false", "gamerule mobGriefing false",
        "gamerule doImmediateRespawn false", "gamerule keepInventory true",
        "gamerule sendCommandFeedback false", "difficulty normal",
        "time set midnight", "weather clear",
        f"fill {x-r} 64 {z-r} {x+r} 70 {z+r} minecraft:air",
        f"fill {x-r} 64 {z-r} {x+r} 64 {z+r} minecraft:bedrock",
        f"fill {x-r} 65 {z-r} {x-r} 69 {z+r} minecraft:bedrock",
        f"fill {x+r} 65 {z-r} {x+r} 69 {z+r} minecraft:bedrock",
        f"fill {x-r} 65 {z-r} {x+r} 69 {z-r} minecraft:bedrock",
        f"fill {x-r} 65 {z+r} {x+r} 69 {z+r} minecraft:bedrock",
        f"spawnpoint @p {x} 65 {z}", f"tp @p {float(x)} 65.0 {float(z)} 0 0",
    ]


def read_state(observation):
    full = observation["full"]
    # Missing counters are a configuration/integration error, not a zero reward.
    if not {"damage_dealt", "damage_taken"} <= full.misc_statistics.keys():
        raise RuntimeError("CraftGround did not supply the requested damage statistics")
    if "zombie" not in full.killed_statistics or ENTITY_RANGE not in full.surrounding_entities:
        raise RuntimeError("CraftGround did not supply zombie statistics/entity observations")
    entities = [Entity(e.unique_name, e.x, e.y, e.z, e.health)
                for e in full.surrounding_entities[ENTITY_RANGE].entities
                if e.translation_key == "entity.minecraft.zombie" and e.health > 0]
    return State(x=full.x, y=full.y, z=full.z, yaw=full.yaw, pitch=full.pitch,
                 health=full.health, is_dead=full.is_dead, entities=entities,
                 damage_dealt=full.misc_statistics["damage_dealt"] / 10.0,
                 damage_taken=full.misc_statistics["damage_taken"] / 10.0,
                 kills=full.killed_statistics["zombie"])


class CombatEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(self, config, condition="B", evaluation=False, port=8000,
                 render_mode=None, backend=None):
        if condition not in ("A", "B", "C"):
            raise ValueError("condition must be A, B, or C")
        self.config, self.condition = config, condition
        self.evaluation, self.port, self.render_mode = evaluation, port, render_mode
        self.settings = config["environment"]
        self.action_space = gym.spaces.MultiDiscrete(COMPACT_NVECS)
        self.history = deque(maxlen=self.settings["history"])
        size = self.settings["history"] * (FRAME_SIZE + len(COMPACT_NVECS))
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (size,), dtype=np.float32)
        self._raw = backend  # Injected only by the lightweight state-machine test.
        self.training_step = 0
        self._done = True
        self.state = None

    def _make_backend(self):
        import craftground
        from craftground import ActionSpaceVersion, InitialEnvironmentConfig
        from craftground.initial_environment_config import Difficulty, GameMode, WorldType
        e = self.settings
        initial = InitialEnvironmentConfig(
            image_width=e["image_width"], image_height=e["image_height"],
            gamemode=GameMode.SURVIVAL, difficulty=Difficulty.NORMAL,
            world_type=WorldType.SUPERFLAT, seed=str(e["world_seed"]),
            generate_structures=False, initial_extra_commands=arena_commands(self.config),
            killed_stat_keys=["zombie"], misc_stat_keys=["damage_dealt", "damage_taken"],
            surrounding_entity_distances=[ENTITY_RANGE], render_distance=2,
            simulation_distance=5, hud_hidden=True, no_fov_effect=True,
        )
        return craftground.make(initial_env_config=initial, mc_version="1.21",
                                action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
                                port=self.port, use_vglrun=e["use_vglrun"])

    def set_training_step(self, step):
        self.training_step = int(step)

    def _wave_limit(self):
        if self.evaluation:
            return self.settings["max_waves"]
        limit = 1
        for start, waves in self.settings["curriculum"]:
            if self.training_step >= start:
                limit = waves
        return min(limit, self.settings["max_waves"])

    def _spawn_commands(self, wave):
        e = self.settings
        encounter = None
        # Training-only scaffolding for the first encounter. Evaluation always
        # retains the original full spawn range and random orientation.
        if not self.evaluation and wave == 1:
            for stage in e.get("encounter_curriculum", []):
                if self.training_step >= stage[0]:
                    encounter = stage
        near, far = min(4.0, e["arena_radius"] - 2), e["arena_radius"] - 2
        if encounter is not None:
            _, near, far, _ = encounter
        positions = []
        for _ in range(wave):
            for _ in range(1000):
                radius = self.np_random.uniform(near, far)
                angle = self.np_random.uniform(-math.pi, math.pi)
                x, z = radius * math.cos(angle), radius * math.sin(angle)
                if all(math.hypot(x-a, z-b) >= 1.5 for a, b in positions):
                    positions.append((x, z))
                    break
            else:
                raise RuntimeError("Could not sample separated enemy spawns")
        commands = [f'summon minecraft:zombie {x+e["arena_center_x"]:.3f} 65.0 '
                f'{z+e["arena_center_z"]:.3f} '
                '{IsBaby:0b,PersistenceRequired:1b,CanPickUpLoot:0b,'
                'HandItems:[{},{}],ArmorItems:[{},{},{},{}],Tags:["cs557_enemy"]}'
                for x, z in positions]
        if encounter is not None and encounter[3] < 180:
            x, z = positions[0]
            yaw = math.degrees(math.atan2(-x, z)) + self.np_random.uniform(-encounter[3], encounter[3])
            pitch = math.degrees(math.atan2(0.62, math.hypot(x, z)))
            commands.insert(0, f'tp @p {float(e["arena_center_x"])} 65.0 '
                            f'{float(e["arena_center_z"])} {yaw:.3f} {pitch:.3f}')
        return commands

    def _loadout_commands(self, yaw):
        e = self.settings
        commands = ["kill @e[type=!minecraft:player]", "gamemode survival @p", "clear @p",
                    "effect clear @p", "effect give @p minecraft:instant_health 1 10 true",
                    "effect give @p minecraft:saturation infinite 255 true",
                    f'tp @p {float(e["arena_center_x"])} 65.0 {float(e["arena_center_z"])} {yaw:.3f} 0',
                    f'item replace entity @p weapon.mainhand with minecraft:{e["weapon"]}[minecraft:unbreakable={{}}] 1']
        if e["armor"] != "none":
            for slot, item in [("head", "helmet"), ("chest", "chestplate"),
                               ("legs", "leggings"), ("feet", "boots")]:
                commands.append(f'item replace entity @p armor.{slot} with '
                                f'minecraft:{e["armor"]}_{item}[minecraft:unbreakable={{}}] 1')
        return commands

    def _observation(self, action):
        frame = encode_frame(self.state, self.wave, self.config, self.condition == "C")
        self.history.append(np.concatenate((frame, normalize_action(action))))
        return np.concatenate(list(self.history)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._raw is None:
            self._raw = self._make_backend()
        self.wave, self.waves_completed = 1, 0
        self.wave_target = self._wave_limit()
        self.elapsed_steps = self.elapsed_ticks = 0
        self.totals = {"kills": 0, "damage_dealt": 0.0, "damage_taken": 0.0}
        self._raw.reset(options={"fast_reset": True})
        # extra_commands on reset are ignored on first launch in CraftGround 2.7.4.
        # Use the same command path on first launch and subsequent fast resets.
        yaw = self.np_random.uniform(-180, 180)
        noop = action_to_craftground(NO_OP_ACTION)
        self._raw.add_commands(self._loadout_commands(yaw))
        # Fast reset kills the player. Let asynchronous statistic updates settle
        # before spawning enemies, so respawn damage never earns a combat penalty.
        previous_counters = None
        stable_ticks = 0
        for _ in range(40):
            observation, *_ = self._raw.step(noop)
            setup = read_state(observation)
            counters = (setup.kills, setup.damage_dealt, setup.damage_taken)
            stable_ticks = stable_ticks + 1 if counters == previous_counters else 0
            previous_counters = counters
            if stable_ticks >= 5 and not setup.entities and setup.health >= 19.99 and not setup.is_dead:
                break
        else:
            raise RuntimeError("Reset cleanup/statistic counters did not settle")
        self._raw.add_commands(["effect clear @p minecraft:instant_health"] + self._spawn_commands(1))
        for _ in range(40):
            observation, *_ = self._raw.step(noop)
            self.state = read_state(observation)
            e = self.settings
            centered = math.hypot(self.state.x-e["arena_center_x"],
                                  self.state.z-e["arena_center_z"]) < 0.5
            if (not self.state.is_dead and self.state.health >= 19.99 and centered
                    and len(self.state.entities) == 1):
                break
        else:
            raise RuntimeError("Reset failed: expected full health, center position, and one zombie")
        self._wave_kills_start = self.state.kills
        self._attack_held = False
        self._done = False
        self.terminal_reason = ""
        self.history.clear()
        for _ in range(self.history.maxlen):
            obs = self._observation(NO_OP_ACTION)
        return obs, self._info()

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a finished episode")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid compact combat action: {action}")
        command = action_to_craftground(action)
        total_reward = 0.0
        terminated = False
        for tick in range(self.settings["action_repeat"]):
            tick_action = dict(command)
            if tick:
                tick_action["camera_yaw"] = tick_action["camera_pitch"] = 0.0
            # CraftGround mouse buttons are edge-triggered: release between clicks.
            tick_action["attack"] = int(tick == 0 and command["attack"] and not self._attack_held)
            self._attack_held = bool(tick_action["attack"])
            raw, *_ = self._raw.step(tick_action)
            previous, self.state = self.state, read_state(raw)
            self.elapsed_ticks += 1
            dead = self.state.is_dead or self.state.health <= 0
            cleared = (not dead and not self.state.entities
                       and self.state.kills - self._wave_kills_start >= self.wave)
            value, deltas = reward(previous, self.state, cleared, dead, self.config,
                                   shaped=self.condition != "A", elapsed_ticks=1)
            total_reward += value
            for key in self.totals:
                self.totals[key] += deltas[key]
            if dead:
                terminated, self.terminal_reason = True, "death"
            elif cleared:
                self.waves_completed += 1
                if self.waves_completed == self.wave_target:
                    terminated, self.terminal_reason = True, "success"
                else:
                    self.wave += 1
                    self._wave_kills_start = self.state.kills
                    self._raw.add_commands(self._spawn_commands(self.wave))
            if terminated:
                break
        self.elapsed_steps += 1
        if not self.evaluation:
            # DummyVecEnv resets immediately after step(), before the callback.
            self.training_step += 1
        truncated = not terminated and self.elapsed_steps >= self.settings["max_episode_steps"]
        if truncated:
            self.terminal_reason = "time_limit"
        self._done = terminated or truncated
        return self._observation(action), total_reward, terminated, truncated, self._info()

    def _info(self):
        return dict(self.totals, waves_completed=self.waves_completed,
                    success=self.terminal_reason == "success", wave_target=self.wave_target,
                    elapsed_steps=self.elapsed_steps, elapsed_seconds=self.elapsed_ticks / 20,
                    terminal_reason=self.terminal_reason)

    def render(self):
        return self._raw.render() if self._raw is not None else None

    def close(self):
        if self._raw is not None:
            self._raw.close()
            self._raw = None
