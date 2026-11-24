"""Matched, inference-only behavior traces; never inject diagnostic state into PPO."""
import argparse
import csv
import faulthandler
import gzip
import json
import math
from pathlib import Path
import time

from .behavior import BehaviorTracker, DEFINITIONS, aggregate
from .evaluate import scenario_start
from .survival_task import make_environment, validate_spec
from .task import NO_OP_ACTION
from .telemetry import CombatTelemetry
from .train import read_checkpoint, runtime_metadata
from scripts.queue_optimization_windows import sha256, write_json


class BehaviorTap:
    def __init__(self, backend):
        self.backend,self.latest,self.behavior,self.crit = backend,None,None,None
    def __getattr__(self,name):
        return getattr(self.backend,name)
    def reset(self,**kwargs):
        self.behavior = self.crit = None
        result = self.backend.reset(**kwargs)
        self.latest = result[0]
        return result
    def begin(self,config,condition,emit):
        self.behavior = BehaviorTracker(config,condition,emit)
        self.behavior.begin(self.latest['full'])
        self.crit = CombatTelemetry()
        self.crit.begin(self.latest['full'])
    def step(self,action):
        result = self.backend.step(action)
        self.latest = result[0]
        if self.behavior is not None:
            critical = self.crit.observe(self.latest['full'],action)
            self.behavior.observe(self.latest['full'],action,critical)
        return result


def evaluate(args):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    model_path, previous = read_checkpoint(args.model)
    spec = validate_spec(json.loads(args.task_spec.read_text()))
    config,condition = previous['config'],previous['condition']
    first = scenario_start(config,'validation',args.episodes,args.first_seed)
    model_hash,meta_hash = sha256(model_path),sha256(model_path.with_suffix('.json'))
    if args.output.exists():
        raise ValueError('Use a new behavior output directory')
    model = PPO.load(model_path,device='cpu') if args.policy=='ppo' else None
    if model is not None:
        model.set_random_seed(first)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    env = make_environment(config,condition,spec,args.port)
    records = []
    try:
        print('Behavior evaluation: constructing simulator backend',flush=True)
        faulthandler.dump_traceback_later(300)
        try:
            env._raw = BehaviorTap(env._make_backend())
        finally:
            faulthandler.cancel_dump_traceback_later()
        with gzip.open(args.output/'ticks.csv.gz','wt',newline='',encoding='utf-8') as stream:
            writer = None
            for seed in range(first,first+args.episodes):
                print(f'Behavior evaluation: resetting scenario {seed}',flush=True)
                faulthandler.dump_traceback_later(300)
                try:
                    observation,_ = env.reset(seed=seed)
                finally:
                    faulthandler.cancel_dump_traceback_later()
                env.action_space.seed(seed)
                def emit(row):
                    nonlocal writer
                    row = dict(scenario_seed=seed,**row)
                    if writer is None:
                        writer = csv.DictWriter(stream,fieldnames=row)
                        writer.writeheader()
                    writer.writerow(row)
                env._raw.begin(config,condition,emit)
                while True:
                    if model is not None:
                        action,_ = model.predict(observation,deterministic=True)
                    elif args.policy=='random':
                        action = env.action_space.sample()
                    else:
                        action = NO_OP_ACTION.copy()
                        action[-1] = 1  # Explicit steady-attack reference, not a reconstructed PPO checkpoint.
                    observation,_,done,truncated,info = env.step(action)
                    if done or truncated:
                        break
                result = env._raw.behavior.finish()
                if result['counts']['ticks'] != round(info['elapsed_seconds']*20):
                    raise RuntimeError('Behavior trace/tick mismatch')
                if (result['focus_fire']['counts']['ticks'] != result['counts']['ticks']
                        or result['focus_fire']['counts'].get('player_kills',0) != info['kills']):
                    raise RuntimeError('Focus trace/episode tick or kill-counter mismatch')
                for key in ('damage_dealt','damage_taken'):
                    if not math.isclose(result['counts'].get(key,0),info[key],abs_tol=1e-5):
                        raise RuntimeError('Behavior trace/episode damage mismatch')
                result.update(scenario_seed=seed,metrics=info,critical_telemetry=env._raw.crit.summary())
                records.append(result)
                write_json(args.output/'episodes.json',records)
                stream.flush()
                print(json.dumps(dict(seed=seed,kills=info['kills'],seconds=info['elapsed_seconds'],
                    retreat_windows=aggregate([result])['movement_windows']['retreat'])),flush=True)
        if sha256(model_path)!=model_hash or sha256(model_path.with_suffix('.json'))!=meta_hash:
            raise RuntimeError('Checkpoint changed during inference')
        summary = aggregate(records)
        summary['mean_kills'] = sum(r['metrics']['kills'] for r in records)/len(records)
        summary['mean_seconds'] = sum(r['metrics']['elapsed_seconds'] for r in records)/len(records)
        write_json(args.output/'summary.json',summary)
        write_json(args.output/'metadata.json',dict(policy=args.policy,model=str(model_path.resolve()),
            model_sha256=model_hash,model_metadata_sha256=meta_hash,model_steps=previous['timesteps'],
            config=config,condition=condition,task=spec,training_task=previous.get('task_spec',{'mode':'waves'}),
            first_seed=first,episodes=args.episodes,split='validation',deterministic=args.policy!='random',
            policy_updated=False,definitions=DEFINITIONS,runtime=runtime_metadata(),
            wall_seconds=time.monotonic()-started,trace_sha256=sha256(args.output/'ticks.csv.gz'),
            note='Fresh matched rollouts; simulator is not bit-exact. All diagnostic ground truth is passive. '
                 'Reference policies use checkpoint settings only; neither is a saved initial PPO policy.'))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--task-spec',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--policy',choices=['ppo','random','steady'],default='ppo')
    parser.add_argument('--episodes',type=int,default=10)
    parser.add_argument('--first-seed',type=int,default=53000)
    parser.add_argument('--port',type=int,default=8000)
    evaluate(parser.parse_args())


if __name__=='__main__':
    main()
