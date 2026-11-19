"""Small, simulator-independent pieces of the combat task."""

from dataclasses import dataclass, field
import math

import numpy as np


MAX_ENEMIES = 5
FRAME_SIZE = 7 + 5 * MAX_ENEMIES
COMPACT_NVECS = np.array([3, 3, 5, 3, 2, 2, 2], dtype=np.int64)
NO_OP_ACTION = np.array([0, 0, 2, 1, 0, 0, 0], dtype=np.int64)
YAW_DEGREES = (-30.0, -10.0, 0.0, 10.0, 30.0)
PITCH_DEGREES = (-10.0, 0.0, 10.0)


@dataclass
class Entity:
    uuid: str
    x: float
    y: float
    z: float
    health: float


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    health: float = 20.0
    is_dead: bool = False
    entities: list[Entity] = field(default_factory=list)
    # Cumulative player statistics. Damage is health points, not Minecraft's
    # integer statistic units (the simulator adapter divides those by ten).
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    kills: int = 0


def encode_frame(state, wave, config, restricted=False):
    """Return bounded player features and five nearest live enemy slots.

    Minecraft yaw zero faces +Z; positive yaw turns toward -X. Each enemy
    slot is [right, forward, up, health, present]. Empty slots contain zeros.
    Restricted observations apply a horizontal field of view, without ray
    tracing. Hidden entities are removed before sorting and never affect
    visible slots. Enemy counts and cumulative reward statistics are omitted.
    """
    env = config["environment"]
    radius = float(env["arena_radius"])
    if radius <= 0:
        raise ValueError("arena_radius must be positive")
    yaw = math.radians(state.yaw)
    sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)
    features = np.zeros(FRAME_SIZE, dtype=np.float32)
    features[:7] = [
        state.health / 20.0,
        (state.x - env.get("arena_center_x", 0.0)) / radius,
        (state.z - env.get("arena_center_z", 0.0)) / radius,
        sin_yaw,
        cos_yaw,
        state.pitch / 90.0,
        wave / MAX_ENEMIES,
    ]
    visible = []
    half_fov = math.radians(env.get("fov_degrees", 100.0)) / 2
    for entity in state.entities:
        if entity.health <= 0:
            continue
        dx, dy, dz = entity.x - state.x, entity.y - state.y, entity.z - state.z
        right = -cos_yaw * dx - sin_yaw * dz
        forward = -sin_yaw * dx + cos_yaw * dz
        if restricted and abs(math.atan2(right, forward)) > half_fov:
            continue
        visible.append((dx * dx + dy * dy + dz * dz, entity.uuid, right, forward, dy, entity.health))
    visible.sort(key=lambda item: (item[0], item[1]))
    for index, (_, _, right, forward, up, health) in enumerate(visible[:MAX_ENEMIES]):
        start = 7 + 5 * index
        features[start:start + 5] = [right / (2 * radius), forward / (2 * radius), up / radius, health / 20.0, 1.0]
    return np.clip(features, -1.0, 1.0)


def _validated_action(action):
    values = np.asarray(action)
    if values.shape != COMPACT_NVECS.shape:
        raise ValueError("Expected seven compact action choices")
    if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
        raise ValueError("Action choices must be finite integers")
    if np.any(values < 0) or np.any(values >= COMPACT_NVECS):
        raise ValueError("Compact action choice is out of range")
    return values.astype(np.int64)


def normalize_action(action):
    """Encode previous categorical choices as seven bounded history features."""
    values = _validated_action(action)
    return (2.0 * values / (COMPACT_NVECS - 1) - 1.0).astype(np.float32)


def action_to_craftground(action):
    """Translate the compact action into CraftGround's complete V2 action dict.

    Dimensions: move (still/forward/back), strafe (still/left/right), yaw,
    pitch, jump, sprint, attack. Camera changes are degrees for one game tick;
    the environment applies them only once when repeating an action.
    """
    move, strafe, yaw, pitch, jump, sprint, attack = _validated_action(action)
    result = dict.fromkeys(
        ("attack", "back", "forward", "jump", "left", "right", "sneak",
         "sprint", "use", "drop", "inventory"), False
    )
    result.update({f"hotbar.{index}": False for index in range(1, 10)})
    result.update(
        forward=bool(move == 1), back=bool(move == 2),
        left=bool(strafe == 1), right=bool(strafe == 2),
        jump=bool(jump), sprint=bool(sprint), attack=bool(attack),
        camera_yaw=YAW_DEGREES[yaw], camera_pitch=PITCH_DEGREES[pitch],
    )
    return result


def reward(prev, state, wave_completed, died, config, shaped=True, elapsed_ticks=1):
    """Reward player-attributed statistic deltas, outcomes, and elapsed ticks.

    Counter resets never produce negative damage or kills. Setup/respawn
    observations must become the new baseline in the environment, rather than
    being rewarded. Death is an event supplied by the environment, so its
    penalty is applied exactly once at termination.
    """
    if elapsed_ticks < 0:
        raise ValueError("elapsed_ticks must be nonnegative")
    weights = config["reward"]
    deltas = {
        "kills": max(0, int(state.kills - prev.kills)),
        "damage_dealt": max(0.0, state.damage_dealt - prev.damage_dealt),
        "damage_taken": max(0.0, state.damage_taken - prev.damage_taken),
    }
    value = (
        weights["kill"] * deltas["kills"]
        + weights["wave"] * int(wave_completed)
        + weights["death"] * int(died)
        + weights["time"] * elapsed_ticks
    )
    if shaped:
        value += weights["damage_dealt"] * deltas["damage_dealt"] / 20.0
        value -= weights["damage_taken"] * deltas["damage_taken"] / 20.0
    return float(value), deltas
