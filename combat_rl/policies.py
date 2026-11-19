"""A simple controller using exactly the same observation as PPO."""

import math

import numpy as np

from combat_rl.task import FRAME_SIZE, NO_OP_ACTION, PITCH_DEGREES, YAW_DEGREES


def latest_frame(observation):
    """History is oldest first; each chunk is a frame followed by an action."""
    return np.asarray(observation)[-(FRAME_SIZE + 7):-7]


def choose_action(observation, step, config):
    frame = latest_frame(observation)
    action = NO_OP_ACTION.copy()
    enemies = frame[7:].reshape(-1, 5)
    enemies = enemies[enemies[:, 4] > 0.5]
    if len(enemies) == 0:
        action[2] = 3  # Search by turning ten degrees right.
        return action
    radius = config["environment"]["arena_radius"]
    enemy = min(enemies, key=lambda row: row[0] ** 2 + row[1] ** 2)
    right, forward, up = enemy[:3] * [2 * radius, 2 * radius, radius]
    distance = math.hypot(right, forward)
    bearing = math.degrees(math.atan2(right, forward))
    action[2] = int(np.argmin(np.abs(np.asarray(YAW_DEGREES) - bearing)))
    # Aim roughly at a zombie's torso from a standing player's eye height.
    pitch = math.degrees(math.atan2(1.62 - (up + 1.0), max(distance, 0.01)))
    pitch_error = pitch - frame[5] * 90
    action[3] = int(np.argmin(np.abs(np.asarray(PITCH_DEGREES) - pitch_error)))
    if abs(bearing) < 60:
        action[0] = 1 if distance > 2.2 else (2 if distance < 1.5 else 0)
    repeat = config["environment"]["action_repeat"]
    cooldown = max(1, math.ceil(13 / repeat))
    action[6] = int(distance < 3 and abs(bearing) < 20 and step % cooldown == 0)
    return action
