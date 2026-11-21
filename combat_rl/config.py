"""Shared experiment settings. A JSON file may override individual defaults."""

import copy
import json
import math
from pathlib import Path

DEFAULTS = {
    "environment": {
        "arena_radius": 12,
        "arena_center_x": 0,
        "arena_center_z": 0,
        "max_waves": 5,
        "max_episode_steps": 3000,
        "action_repeat": 4,
        "history": 4,
        "fov_degrees": 100,
        "image_width": 160,
        "image_height": 90,
        "world_seed": 557,
        "weapon": "iron_sword",
        "armor": "iron",
        "use_vglrun": False,
        "curriculum": [[0, 1], [100000, 3], [300000, 5]],
        "encounter_curriculum": [],
    },
    "reward": {"kill": 1.0, "wave": 3.0, "death": -5.0, "time": -0.001,
               "damage_dealt": 1.0, "damage_taken": 1.5},
    "training": {
        "total_timesteps": 1000000,
        "n_envs": 1,
        "n_steps": 1024,
        "batch_size": 256,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "n_epochs": 10,
        "policy_layers": [128, 128],
        "checkpoint_every": 50000,
        "torch_threads": 1,
        "initial_action_bias": 0.0,
    },
    "evaluation": {"episodes": 50, "validation_seed": 50000, "test_seed": 100000},
}


def load_config(path=None):
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        overrides = json.loads(Path(path).read_text())
        for section, values in overrides.items():
            if section not in config or not isinstance(values, dict):
                raise ValueError(f"Unknown config section: {section}")
            unknown = values.keys() - config[section].keys()
            if unknown:
                raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
            config[section].update(values)
    e = config["environment"]
    for key in ("arena_radius", "arena_center_x", "arena_center_z", "max_waves",
                "max_episode_steps", "action_repeat", "history", "world_seed"):
        if not isinstance(e[key], int):
            raise ValueError(f"{key} must be an integer")
    if not 5 <= e["arena_radius"] <= 20:
        raise ValueError("arena_radius must be in [5, 20] (64-block observation range)")
    if not 1 <= e["max_waves"] <= 5:
        raise ValueError("max_waves must be in [1, 5]")
    if min(e["action_repeat"], e["history"], e["max_episode_steps"]) < 1:
        raise ValueError("action_repeat, history, and max_episode_steps must be positive")
    if not 0 < e["fov_degrees"] <= 360:
        raise ValueError("fov_degrees must be in (0, 360]")
    stages = e["curriculum"]
    if not stages or stages[0][0] != 0 or stages != sorted(stages):
        raise ValueError("curriculum must start at step zero and be sorted")
    if any(step < 0 or not 1 <= waves <= 5 for step, waves in stages):
        raise ValueError("invalid curriculum stage")
    encounters = e["encounter_curriculum"]
    previous_step = -1
    for stage in encounters:
        if not isinstance(stage, list) or len(stage) != 4:
            raise ValueError("encounter stages must be [step, min_radius, max_radius, yaw_error]")
        step, near, far, yaw_error = stage
        if (not isinstance(step, int) or step <= previous_step or step < 0
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in stage)
                or not 2.5 <= near <= far <= e["arena_radius"] - 2
                or not 0 <= yaw_error <= 180):
            raise ValueError("invalid encounter curriculum stage")
        previous_step = step
    if encounters and encounters[0][0] != 0:
        raise ValueError("encounter curriculum must start at step zero")
    action_bias = config["training"]["initial_action_bias"]
    if not isinstance(action_bias, (int, float)) or not math.isfinite(action_bias) or action_bias < 0:
        raise ValueError("initial_action_bias must be finite and nonnegative")
    if e["weapon"] not in ("wooden_sword", "stone_sword", "iron_sword", "diamond_sword", "netherite_sword"):
        raise ValueError("weapon must be a supported sword")
    if e["armor"] not in ("none", "leather", "chainmail", "iron", "diamond", "netherite"):
        raise ValueError("unsupported armor material")
    return config
