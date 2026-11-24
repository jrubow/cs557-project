"""Audit a finished final checkpoint and complete a failed batch queue without retraining.

A missing supervisor receipt stays missing: checkpoint adoption is explicit and
does not assert a clean prior process exit. Original outputs are never rewritten.
"""
import argparse
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys
import zipfile

import psutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.queue_endless_windows import Worker as BaseWorker, PROFILES, improves, summarize
from scripts.queue_optimization_windows import read_json,write_json,sha256,clean_supervisor


def inspect_final(model, parent, task, initial_steps, revision):
    meta,previous = read_json(model.with_suffix('.json')),read_json(parent.with_suffix('.json'))
    config = copy.deepcopy(previous['config'])
    target = initial_steps+2_000_000
    config['training'].update(total_timesteps=target,learning_rate=.0001)
    if (meta['config']!=config or meta['task_spec']!=task or meta['previous_task']!=task
            or meta['condition']!=previous['condition'] or meta['seed']!=previous['seed']
            or Path(meta['resume_from']).resolve()!=parent.resolve() or meta['resume_sha256']!=sha256(parent)
            or meta['task_start_timesteps']!=initial_steps
            or meta['runtime']['git_revision']!=revision or meta['runtime']['git_status']
            or not target<=meta['timesteps']<target+config['training']['n_steps']):
        raise RuntimeError('Final checkpoint provenance/config/budget mismatch')
    with zipfile.ZipFile(model) as archive:
        if archive.testzip() is not None or json.loads(archive.read('data'))['num_timesteps']!=meta['timesteps']:
            raise RuntimeError('Final checkpoint archive/metadata mismatch')
    return meta


def require_idle(old_pids):
    if any(psutil.pid_exists(pid) for pid in old_pids):
        raise RuntimeError('A prior controller is still alive; no handoff permitted')
    for p in psutil.process_iter(['name','cmdline']):
        try:
            if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe','java.exe'):
                cmd=' '.join(p.info['cmdline'] or [])
                if any(key in cmd for key in ('craftground_runtime_mc121','combat_rl.survival_task','scripts/windows_run.py')):
                    raise RuntimeError('An existing simulator/job is still running')
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pass
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',8000))==0:
            raise RuntimeError('Simulator port is occupied')


class Worker(BaseWorker):
    def __init__(self,args):
        self.args,self.source = args,Path(__file__).resolve().parents[1]
        self.root,self.predecessor = args.output.resolve(),args.predecessor.resolve()
        self.revision = self.git('rev-parse','HEAD')
        self.predecessor_hash = sha256(self.predecessor/'manifest.json')
        self.models,self.specs = {},{}
        self.state = dict(jobs_completed=0,overall_deadline=None)
        self.check_source()

    def predecessor_alive(self):
        # audit() requires their exit before any jobs. A later reuse of these
        # numeric PIDs is not a resurrected predecessor.
        return False

    def audit(self):
        require_idle(self.args.old_pids)
        old = self.predecessor
        manifest,state = read_json(old/'manifest.json'),read_json(old/'status.json')
        if state['phase']!='failed' or state['current_job']!='train-additional-2000000':
            raise RuntimeError('This recovery only adopts the finished final training job')
        scientific = manifest['source_revision']
        for name in ('survival_task.py','endless.py','env.py','task.py','train.py','config.py'):
            if self.git('rev-parse',f'{scientific}:combat_rl/{name}')!=self.git('rev-parse',f'HEAD:combat_rl/{name}'):
                raise RuntimeError('Scientific training/evaluation code changed')
        events=[json.loads(line) for line in (old/'ledger.jsonl').read_text().splitlines()]
        started={e['data']['label']:e['data'] for e in events if e['event']=='job_started'}
        completed={e['data']['label']:e['data'] for e in events if e['event']=='job_completed'}
        if (len(started)!=27 or len(completed)!=26 or state['jobs_completed']!=26
                or set(started)-set(completed)!={'train-additional-2000000'}):
            raise RuntimeError('Unexpected prior ledger; explicit review required')
        for label,event in completed.items():
            receipt=clean_supervisor(old/'logs'/(label+'.status.json'))
            command=started[label]['command']
            if receipt!=event['supervisor'] or receipt['command'][1:]!=command[command.index('--')+1:]:
                raise RuntimeError('Prior supervisor identity changed')
        if (old/'logs/train-additional-2000000.status.json').exists():
            raise RuntimeError('Expected missing final supervisor receipt; reassess new evidence')
        selection=read_json(old/'selection.json')
        self.selection=selection
        self.parent=self.remember(selection['parent'])
        self.task=selection['task']
        baseline=summarize(old/'evaluations/baseline-batch.csv',self.parent,self.task,52100,20,scientific)
        if baseline!=selection['baseline'] or selection['held_out_test_used']:
            raise RuntimeError('Baseline or split changed')
        if [c['additional_target'] for c in selection['candidates']]!=[250000,500000,1000000]:
            raise RuntimeError('Unexpected prior candidate sequence')
        for c in selection['candidates']:
            model=self.remember(c['model'])
            score=summarize(old/'evaluations'/f"candidate-{c['additional_target']}.csv",model,self.task,52100,20,scientific)
            if score!=c['score'] or sha256(model)!=c['sha256']:
                raise RuntimeError('Prior candidate identity changed')
        best=self.remember(selection['selected'])
        if sha256(best)!=selection['selected_sha256']:
            raise RuntimeError('Prior selected model changed')
        self.final=self.remember(old/'models/additional_2000000/final.zip')
        previous=Path(selection['candidates'][-1]['model'])
        initial=read_json(self.parent.with_suffix('.json'))['timesteps']
        meta=inspect_final(self.final,previous,self.task,initial,scientific)
        command=started['train-additional-2000000']['command']
        arguments=command[command.index('--')+1:]
        expected=['-m','combat_rl.survival_task','--operation','train','--model',str(previous),
                  '--task-spec',arguments[arguments.index('--task-spec')+1],'--output',str(self.final.parent),
                  '--additional-steps',str(initial+2000000-read_json(previous.with_suffix('.json'))['timesteps'])]
        if arguments!=expected or read_json(Path(expected[7]))!=self.task:
            raise RuntimeError('Final training command/task changed')
        import torch
        from stable_baselines3 import PPO
        model=PPO.load(self.final,device='cpu')
        if model.num_timesteps!=meta['timesteps'] or not all(torch.isfinite(t).all() for t in model.policy.state_dict().values()):
            raise RuntimeError('Final PPO model load/weights invalid')
        history=read_json(old/'critical-history.json')
        for c in history['checkpoints']:
            if sha256(self.remember(c['model']))!=c['sha256']:
                raise RuntimeError('Historical model changed')
        return dict(prior_completed_jobs_verified=26,adopted_model=str(self.final),
            model_sha256=sha256(self.final),metadata_sha256=sha256(self.final.with_suffix('.json')),
            timesteps=meta['timesteps'],prior_supervisor_exit_code=None,prior_cleanup_receipt=None,
            prior_clean_exit_certified=False,current_runtime_idle_verified=True,ppo_load_and_finite_weights=True,
            original_source_revision=scientific,
            evidence_sha256={name:sha256(old/name) for name in ('manifest.json','status.json','ledger.jsonl',
                'selection.json','calibration.json','critical-history.json','logs/train-additional-2000000.log')},
            note='Explicit adoption of a complete audited artifact after loss of supervisor receipt. '
                 'No prior exit code or clean shutdown is inferred. No training is repeated.')

    def run(self):
        if self.root.exists():
            raise RuntimeError('Use fresh recovery outputs')
        audit=self.audit()
        self.root.mkdir(parents=True)
        write_json(self.root/'manifest.json',dict(source_revision=self.revision,predecessor=str(self.predecessor),
            predecessor_manifest_sha256=self.predecessor_hash,old_pids=self.args.old_pids,
            additional_training_steps=0,held_out_test_used=False,overall_deadline=None,recovery=audit))
        write_json(self.root/'critical-history.json',read_json(self.predecessor/'critical-history.json'))
        write_json(self.root/'selection.json',self.selection)
        self.ledger('checkpoint_adopted',audit=audit)
        self.status('resuming_evaluations')
        score=self.evaluate('candidate-2000000',self.final,self.task,52100,20)
        best,best_score=self.remember(self.selection['selected']),self.selection['selected_score']
        baseline=self.selection['baseline']
        if improves(score,baseline) and (score['mean_kills'],score['mean_seconds'])>(best_score['mean_kills'],best_score['mean_seconds']):
            best,best_score=self.final,score
        selection=dict(self.selection,selected=str(best),selected_sha256=sha256(best),selected_score=best_score,
            candidates=self.selection['candidates']+[dict(model=str(self.final),sha256=sha256(self.final),
                                                        additional_target=2000000,score=score)])
        write_json(self.root/'selection.json',selection)  # Lock before fresh confirmation.
        confirmation={name:self.evaluate('confirmation-'+name,model,self.task,52200,20)
                      for name,model in [('parent',self.parent),('selected',best)]}
        transfer={name:self.evaluate('five-wave-transfer-'+name,model,{'mode':'waves'},51000,50)
                  for name,model in [('parent',self.parent),('selected',best)]}
        timed=self.evaluate('selected-timed2',best,PROFILES['timed2'],52100,20)
        videos=[self.record('parent-batch-52000',self.parent,self.task,52000)]
        videos += [self.record(f'selected-batch-{seed}',best,self.task,seed) for seed in range(52000,52003)]
        videos += [self.record('selected-timed2-52000',best,PROFILES['timed2'],52000)]
        write_json(self.root/'results.json',dict(selected=str(best),selected_sha256=sha256(best),task=self.task,
            calibration=read_json(self.predecessor/'calibration.json')['profiles'],baseline=baseline,
            selected_development=best_score,confirmation=confirmation,
            improvement_confirmed=improves(confirmation['selected'],confirmation['parent']),
            five_wave_transfer=transfer,timed_transfer=timed,videos=videos,recovery=audit,
            note='Same predeclared exploratory protocol; original failure retained; no retraining or test use.'))
        self.status('complete',current_job=None,selected=str(best),review_ready=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','predecessor','video-deps'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--old-pids',required=True,type=int,nargs='+')
    args=parser.parse_args()
    sys.path.insert(0,str(args.video_deps.resolve()))
    worker=Worker(args)
    try:
        worker.run()
    except (Exception,KeyboardInterrupt) as error:
        worker.status('failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
