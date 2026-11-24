"""Explicit task-transfer training/evaluation for the exploratory endless study."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import signal
import time

from combat_rl.env import CombatEnv
from combat_rl.endless import EndlessCombatEnv
from combat_rl.evaluate import scenario_start
from combat_rl.telemetry import TelemetryTap
from combat_rl.train import read_checkpoint, runtime_metadata, save_checkpoint, finetune_ppo_kwargs


def validate_spec(spec):
    if spec.get('mode') not in ('waves', 'timed', 'batch'):
        raise ValueError('Task mode must be waves, timed or batch')
    if spec['mode'] != 'waves':
        allowed = {'mode', 'batch_size', 'spawn_seconds', 'max_seconds'}
        if set(spec)-allowed:
            raise ValueError('Unknown task settings')
        # Constructor also validates tick precision, bounds and mode values.
        from combat_rl.config import load_config
        EndlessCombatEnv(load_config(), **spec)
    elif spec != {'mode': 'waves'}:
        raise ValueError('Original waves task has no demonstration overrides')
    return copy.deepcopy(spec)


def make_environment(config, condition, spec, port=8000):
    if spec['mode'] == 'waves':
        return CombatEnv(config, condition=condition, evaluation=True, port=port)
    return EndlessCombatEnv(config, condition=condition, port=port, **spec)


def evaluate(args, model_path, previous, spec):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise ValueError('Use new evaluation outputs')
    config = previous['config']
    first = scenario_start(config, 'validation', args.episodes, args.first_seed)
    model = PPO.load(model_path, device='cpu')
    model.set_random_seed(first)
    env = make_environment(config, previous['condition'], spec, args.port)
    rows, telemetry = [], []
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        env._raw = TelemetryTap(env._make_backend())
        with args.output.open('x', newline='') as stream:
            writer = None
            for seed in range(first, first+args.episodes):
                obs, _ = env.reset(seed=seed)
                tracker = env._raw.begin()
                episode_return = 0
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, value, done, truncated, info = env.step(action)
                    episode_return += value
                    if done or truncated:
                        break
                row = dict(scenario_seed=seed, model=str(model_path), training_seed=previous['seed'],
                    mode=spec['mode'], kills=info['kills'], elapsed_seconds=info['elapsed_seconds'],
                    final_health=env.state.health, damage_dealt=info['damage_dealt'], damage_taken=info['damage_taken'],
                    batches_completed=info.get('batches_completed', 0), waves_completed=info['waves_completed'],
                    peak_active_enemies=info.get('peak_active_enemies', ''),
                    spawn_commands_issued=info.get('spawn_commands_issued', ''),
                    success=info['success'], terminal_reason=info['terminal_reason'],
                    critical_sound_events=tracker.counts['critical_sound_events'],
                    attack_pulses=tracker.counts['attack_pulses'], damage_events=tracker.counts['damage_events'],
                    descending_attack_pulses=tracker.counts['descending_attack_pulses'],
                    jump_ticks=tracker.counts['jump_ticks'], airborne_ticks=tracker.counts['airborne_ticks'],
                    game_ticks=tracker.tick, episode_return=episode_return)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                telemetry.append(dict(scenario_seed=seed, summary=tracker.summary(), events=tracker.events))
                print(json.dumps(dict(scenario_seed=seed, kills=row['kills'], seconds=row['elapsed_seconds'],
                                      critical_sound_events=row['critical_sound_events'])), flush=True)
        args.output.with_suffix('.events.json').write_text(json.dumps(telemetry)+'\n')
        args.output.with_suffix('.json').write_text(json.dumps(dict(model=str(model_path),
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), config=config,
            training_task=previous.get('task_spec', {'mode':'waves'}), evaluation_task=spec,
            episodes=args.episodes, first_seed=first, split='validation', deterministic=True,
            training_seed=previous['seed'], wall_seconds=time.monotonic()-started,
            runtime=runtime_metadata(), policy_updated=False), indent=2)+'\n')
    finally:
        env.close()


def train(args, model_path, previous, spec):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    if spec['mode'] == 'waves' or not 1 <= args.additional_steps <= 2_000_000:
        raise ValueError('Transfer training requires a survival task and 1-2M additional steps')
    if args.output.exists():
        raise ValueError('Use a new training directory; preserve prior checkpoints')
    if previous.get('task_spec') and previous['task_spec'] != spec:
        raise ValueError('A continuation must preserve its survival task specification')
    args.output.mkdir(parents=True)
    config = copy.deepcopy(previous['config'])
    target = previous['timesteps']+args.additional_steps
    config['training'].update(total_timesteps=target, learning_rate=0.0001)
    metadata = dict(condition=previous['condition'], seed=previous['seed'], config=config,
        task_spec=spec, previous_task=previous.get('task_spec', {'mode':'waves'}), algorithm='PPO', device='cpu',
        resume_from=str(model_path), resume_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        task_start_timesteps=previous.get('task_start_timesteps', previous['timesteps']),
        runtime=runtime_metadata(), note='Explicit exploratory transfer to a different spawning task; not part of the A/B/C study.')
    (args.output/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: Monitor(make_environment(config, previous['condition'], spec, args.port),
        str(args.output/'episodes.monitor.csv'), info_keywords=('kills','elapsed_seconds','batches_completed'))])
    model = None
    last_saved = previous['timesteps']
    class Checkpoints(BaseCallback):
        def _on_step(self):
            return True
        def _on_rollout_start(self):
            nonlocal last_saved
            if self.num_timesteps-last_saved >= 10_000:
                save_checkpoint(self.model, args.output/f'step_{self.num_timesteps}', metadata)
                last_saved = self.num_timesteps
            if self.model.ep_info_buffer:
                for key in ('kills','elapsed_seconds','batches_completed'):
                    self.logger.record('survival/'+key,
                        sum(float(r[key]) for r in self.model.ep_info_buffer)/len(self.model.ep_info_buffer))
    def stop(*_):
        raise KeyboardInterrupt
    old_handler = signal.signal(signal.SIGTERM, stop)
    try:
        env.seed(previous['seed'])
        model = PPO.load(model_path, env=env, device='cpu', **finetune_ppo_kwargs(config))
        if model.ep_info_buffer is not None:
            model.ep_info_buffer.clear()
        if model.ep_success_buffer is not None:
            model.ep_success_buffer.clear()
        model.set_logger(configure(str(args.output/'logs'), ['stdout','csv']))
        model.learn(target-model.num_timesteps, reset_num_timesteps=False, callback=Checkpoints())
        save_checkpoint(model, args.output/'final', metadata)
    except BaseException:
        if model is not None:
            save_checkpoint(model, args.output/'interrupted', metadata)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        try:
            if model is not None and getattr(model, '_logger', None) is not None:
                model.logger.close()
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', required=True, choices=['train','evaluate'])
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--task-spec', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--first-seed', type=int, default=52100)
    parser.add_argument('--additional-steps', type=int, default=500000)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    model, previous = read_checkpoint(args.model)
    spec = validate_spec(json.loads(args.task_spec.read_text()))
    {'train':train, 'evaluate':evaluate}[args.operation](args, model, previous, spec)


if __name__ == '__main__':
    main()
