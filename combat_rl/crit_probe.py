"""Validate critical-sound telemetry with labeled scripted controls, not PPO data."""
import argparse
import copy
import json
import math
from pathlib import Path

from combat_rl.env import CombatEnv
from combat_rl.task import NO_OP_ACTION
from combat_rl.telemetry import TelemetryTap
from combat_rl.train import read_checkpoint, runtime_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path, help='Supply settings only; this probe never loads/trains a policy')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new probe output')
    _, previous = read_checkpoint(args.model)
    config = copy.deepcopy(previous['config'])
    config['environment']['action_repeat'] = 1
    env = CombatEnv(config, evaluation=True, condition='C')
    records = []
    try:
        env._raw = TelemetryTap(env._make_backend())
        for falling in (False, True):
            env.reset(seed=54000+int(falling))
            env._raw.add_commands(['tp @p 0 65 0 0 20',
                'tp @e[tag=cs557_enemy,limit=1] 0 65 2',
                'data merge entity @e[tag=cs557_enemy,limit=1] {NoAI:1b}'])
            for _ in range(30):
                env.step(NO_OP_ACTION)
            tracker = env._raw.begin()
            attacked = False
            for tick in range(45):
                action = NO_OP_ACTION.copy()
                if falling:
                    action[4] = int(tick == 0)
                    full = env._raw.latest['full']
                    target_pitch = math.degrees(math.atan2(env.state.y+1.62-66, 2))
                    error = target_pitch-env.state.pitch
                    action[3] = 2 if error > 5 else 0 if error < -5 else 1
                    action[-1] = int(not attacked and not full.is_on_ground and full.velocity_y < -0.01)
                else:
                    action[-1] = int(tick == 0)
                attacked |= bool(action[-1])
                _, _, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            records.append(dict(control='falling' if falling else 'grounded', attack_issued=attacked,
                                summary=tracker.summary(), events=tracker.events))
        ground, air = records
        validated = (ground['summary']['critical_sound_events'] == 0 and ground['summary']['damage_events'] > 0
                     and air['summary']['critical_sound_events'] >= 1 and air['summary']['damage_events'] > 0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(validated=validated, controls=records, runtime=runtime_metadata(),
            note='Scripted sensor calibration with an immobile zombie, changed positioning and one-tick actions. Not learned-policy performance; no checkpoint updates. If false, sound-cue counts remain unvalidated.'), indent=2)+'\n')
        print(json.dumps({'critical_sound_detector_validated':validated}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
