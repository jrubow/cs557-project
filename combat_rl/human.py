"""Collect a human baseline through the same structured observation and actions."""

import argparse
import csv
import json
import math
from pathlib import Path
import time

from combat_rl.config import load_config
from combat_rl.evaluate import METRICS
from combat_rl.policies import latest_frame
from combat_rl.task import NO_OP_ACTION


def draw(screen, font, observation, episode, condition, pygame):
    frame = latest_frame(observation)
    screen.fill((22, 26, 32))
    center, scale = (350, 330), 280
    pygame.draw.circle(screen, (55, 62, 72), center, scale, 1)
    pygame.draw.line(screen, (55, 62, 72), (70, 330), (630, 330))
    pygame.draw.line(screen, (55, 62, 72), (350, 50), (350, 610))
    corners = []
    for x, z in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
        dx, dz = (x - frame[1]) / 2, (z - frame[2]) / 2
        right, forward = -frame[4] * dx - frame[3] * dz, -frame[3] * dx + frame[4] * dz
        corners.append((int(center[0] + right * scale), int(center[1] - forward * scale)))
    pygame.draw.polygon(screen, (135, 140, 150), corners, 2)
    pygame.draw.polygon(screen, (95, 190, 255), [(350, 318), (342, 340), (358, 340)])
    for right, forward, up, health, present in frame[7:].reshape(-1, 5):
        if present < 0.5:
            continue
        point = (int(center[0] + right * scale), int(center[1] - forward * scale))
        pygame.draw.circle(screen, (225, 85, 75), point, 8)
        pygame.draw.rect(screen, (100, 220, 120), (*[point[0] - 10, point[1] - 15], int(20 * health), 3))
        screen.blit(font.render(f"y/r={up:.2f}", True, (190, 195, 200)), (point[0] + 10, point[1]))
    yaw = math.degrees(math.atan2(frame[3], frame[4]))
    lines = [f"Episode {episode + 1} | condition {condition} | wave {round(frame[6] * 5)} | health {frame[0] * 20:.1f}",
             f"Position ({frame[1]:.2f}, {frame[2]:.2f}) arena radii | yaw {yaw:.0f} | pitch {frame[5] * 90:.0f}",
             "W/S move | A/D strafe | arrows aim (CTRL: fast yaw) | SPACE jump",
             "SHIFT sprint | F attack | ESC quit. Red: observed enemies; green: health."]
    for index, line in enumerate(lines):
        screen.blit(font.render(line, True, (230, 234, 240)), (12, 8 + index * 20))
    pygame.display.flip()


def keyboard_action(keys, pygame):
    action = NO_OP_ACTION.copy()
    action[0] = 1 if keys[pygame.K_w] else (2 if keys[pygame.K_s] else 0)
    action[1] = 1 if keys[pygame.K_a] else (2 if keys[pygame.K_d] else 0)
    action[2] = 1 if keys[pygame.K_LEFT] else (3 if keys[pygame.K_RIGHT] else 2)
    if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
        action[2] = 0 if keys[pygame.K_LEFT] else (4 if keys[pygame.K_RIGHT] else 2)
    action[3] = 0 if keys[pygame.K_UP] else (2 if keys[pygame.K_DOWN] else 1)
    action[4] = keys[pygame.K_SPACE]
    action[5] = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
    action[6] = keys[pygame.K_f]
    return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--condition", choices=["A", "B", "C"], default="B")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--participant", default="anonymous", help="Use an anonymous participant ID")
    args = parser.parse_args()
    if args.episodes < 1 or args.output.exists():
        parser.error("Episodes must be positive and output must not already exist")
    import pygame
    from combat_rl.env import CombatEnv
    config = load_config(args.config)
    first_seed = config["evaluation"][args.split + "_seed"]
    pygame.init()
    screen = pygame.display.set_mode((700, 650))
    pygame.display.set_caption("CS557 structured-observation human baseline")
    font = pygame.font.Font(None, 20)
    interval = config["environment"]["action_repeat"] / 20
    env = None
    completed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["policy", "condition", "training_seed", "model", "split", "scenario_seed",
              "episode_return", *METRICS, "participant"]
    try:
        env = CombatEnv(config, condition=args.condition, evaluation=True, port=args.port, render_mode=None)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for episode in range(args.episodes):
                observation, _ = env.reset(seed=first_seed + episode)
                episode_return = 0.0
                while True:
                    started = time.perf_counter()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            return  # Incomplete episodes are not scored.
                    draw(screen, font, observation, episode, args.condition, pygame)
                    action = keyboard_action(pygame.key.get_pressed(), pygame)
                    observation, value, terminated, truncated, info = env.step(action)
                    episode_return += value
                    # Simulator steps count toward the real-time budget; poll
                    # events again on the next decision, with no input pause.
                    remaining = interval - (time.perf_counter() - started)
                    if remaining > 0:
                        pygame.time.wait(round(remaining * 1000))
                    if terminated or truncated:
                        break
                row = dict(policy="human", condition=args.condition, training_seed=args.participant,
                           model="", split=args.split, scenario_seed=first_seed + episode,
                           episode_return=episode_return, participant=args.participant)
                row.update({key: info[key] for key in METRICS})
                writer.writerow(row)
                stream.flush()
                completed += 1
                print(f"{completed}/{args.episodes}: waves={info['waves_completed']}, success={info['success']}", flush=True)
    finally:
        if env is not None:
            env.close()
        pygame.quit()
        args.output.with_suffix(".json").write_text(json.dumps({
            "policy": "human", "participant": args.participant, "condition": args.condition,
            "split": args.split, "first_seed": first_seed, "episodes_completed": completed,
            "config": config, "interface": "Structured observations, heading-up diagram; contextual baseline."}, indent=2) + "\n")


if __name__ == "__main__":
    main()
