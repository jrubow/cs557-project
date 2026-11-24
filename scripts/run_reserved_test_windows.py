"""Lock development-selected policies, then run the reserved five-wave test once."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import statistics
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.queue_endless_windows import Worker as BaseWorker
from scripts.finish_endless_windows import require_idle
from scripts.queue_optimization_windows import read_json,write_json,sha256
from scripts.run_test_evaluation_windows import validate_result,METRICS

FIRST=200000


def checkpoint_job(name,group,path,seed,target):
    model=Path(path).resolve()
    m=read_json(model.with_suffix('.json'))
    if (m['condition']!='C' or m['seed']!=seed or m['config']['environment']['max_waves']!=5
            or m['runtime']['git_status'] or not target<=m['timesteps']<target+1024):
        raise RuntimeError('Locked checkpoint identity/budget mismatch')
    with zipfile.ZipFile(model) as archive:
        if json.loads(archive.read('data'))['num_timesteps']!=m['timesteps']:
            raise RuntimeError('Checkpoint step mismatch')
    return dict(name=name,group=group,policy='ppo',condition='C',training_seed=seed,
        training_target=target,steps=m['timesteps'],model=str(model),model_sha256=sha256(model),
        metadata_sha256=sha256(model.with_suffix('.json')),training_revision=m['runtime']['git_revision'])


def summarize(records):
    groups=[]
    for group in dict.fromkeys(r['group'] for r in records):
        rows=[r for r in records if r['group']==group]
        out=dict(group=group,training_seeds=len(rows),episodes=sum(r['episodes'] for r in rows),
                 successes=sum(r['successes'] for r in rows))
        for key in METRICS:
            values=[r[key] for r in rows]
            out[key+'_mean']=statistics.mean(values)
            out[key+'_sd']=statistics.stdev(values) if len(values)>1 else None
        groups.append(out)
    paired=[]
    for seed in range(3):
        rows={r['group']:r for r in records if r['training_seed']==seed}
        if {'C 2M parent','survival10 4M'}<=rows.keys():
            paired.append(dict(seed=seed,**{key:rows['survival10 4M'][key]-rows['C 2M parent'][key]
                                            for key in ('success','waves_completed','kills')}))
    return dict(complete=len(records)==7,first_seed=FIRST,last_seed=FIRST+49,runs=records,
        groups=groups,paired_seed_changes=paired,episodes=sum(r['episodes'] for r in records),
        selection_changed=False,policy_updated=False,
        note='Each model uses the same 50 reserved scenarios. Means weight training seeds equally; '
             'SD describes three related training-seed lineages, not 150 independent training runs. '
             'Parent vs survival includes extra training and joint settings changes. '
             'Batch adaptation is a separate single-seed transfer result. No selection or tuning uses these results.')


class Worker(BaseWorker):
    def __init__(self,args):
        self.args,self.source=args,Path(__file__).resolve().parents[1]
        self.root,self.predecessor=args.output.resolve(),args.survival_root.resolve()
        self.revision=self.git('rev-parse','HEAD')
        self.predecessor_hash=sha256(self.predecessor/'manifest.json')
        self.models,self.specs={},{}
        self.state=dict(jobs_completed=0,overall_deadline=None,total_jobs=7,total_episodes=350)
        self.check_source()

    def predecessor_alive(self):
        return False  # Startup and every job independently verify runtime idleness.

    def lock_plan(self):
        require_idle([])
        for root in (self.predecessor,self.args.batch_root,self.args.behavior_root):
            if read_json(root/'status.json')['phase']!='complete':
                raise RuntimeError('Development and behavior queues must be complete')
        selection=read_json(self.predecessor/'selection.json')
        original=read_json(self.predecessor/'manifest.json')
        batch=read_json(self.args.batch_root/'results.json')
        if (selection['status']!='confirmed_improvement' or selection['new_test_used']
                or selection['reserved_test_scenarios']!=[FIRST,FIRST+49]
                or original['reserved_test_scenarios']!=list(range(FIRST,FIRST+50))
                or selection['best_seed0']['recipe']!='survival10' or selection['best_seed0']['target']!=4000000
                or len(selection['retained_models'])!=3 or len(original['parents'])!=3):
            raise RuntimeError('Expected locked, untested three-seed survival10/4M selection')
        jobs=[]
        for seed in range(3):
            parent=original['parents'][seed]
            j=checkpoint_job(f'parent_2M_seed{seed}','C 2M parent',parent['model'],seed,2000000)
            if j['model_sha256']!=parent['model_sha256'] or j['metadata_sha256']!=parent['metadata_sha256']:
                raise RuntimeError('Parent identity changed since survival study')
            jobs.append(j)
            jobs.append(checkpoint_job(f'survival10_4M_seed{seed}','survival10 4M',
                                      selection['retained_models'][seed],seed,4000000))
        batch_meta=read_json(Path(batch['selected']).with_suffix('.json'))
        if (sha256(batch['selected'])!=batch['selected_sha256'] or batch['task']['mode']!='batch'
                or batch_meta['task_spec']!=batch['task']):
            raise RuntimeError('Batch selection identity changed')
        jobs.append(checkpoint_job('batch_plus1M_seed0','batch +1M transfer',batch['selected'],0,
                                   batch_meta['config']['training']['total_timesteps']))
        common=read_json(Path(jobs[0]['model']).with_suffix('.json'))['config']['environment']
        for job in jobs:
            self.remember(job['model'])
            if read_json(Path(job['model']).with_suffix('.json'))['config']['environment']!=common:
                raise RuntimeError('All final tests must preserve the original arena/observation/action task')
        evidence=[self.predecessor/'selection.json',self.predecessor/'manifest.json',
                  self.args.batch_root/'results.json',self.args.behavior_root/'results.json']
        hashes={str(p.resolve()):sha256(p) for p in evidence}
        self.models.update({Path(path):digest for path,digest in hashes.items()})
        return dict(source_revision=self.revision,source=str(self.source),jobs=jobs,
            first_seed=FIRST,last_seed=FIRST+49,episodes_per_model=50,total_episodes=350,
            deterministic=True,overall_deadline=None,policy_updated=False,selection_locked_before_test=True,
            development_evidence_sha256=hashes,
            primary_recipe='survival10 4M',demonstration_model=selection['best_seed0']['model'],
            batch_demo_model=batch['selected'],batch_upgrade_confirmed=batch['improvement_confirmed'],
            rationale='Keep the confirmed three-seed survival10/4M recipe and its development-selected seed-0 demo. '
                      'Keep the batch +1M model as an exploratory demo, not a confirmed replacement. '
                      'Compare each survival seed with its own C/2M parent; report batch transfer separately. '
                      'The smaller behavior sample is diagnostic and does not reselect checkpoints.',
            frozen_utc=datetime.now(timezone.utc).isoformat())

    def run(self):
        if self.root.exists():
            raise RuntimeError('Use a fresh test output; never overwrite held-out results')
        protocol=self.lock_plan()
        marker=self.predecessor/'reserved-test-lock.json'
        # A separate immutable reservation leaves the original development selection intact.
        with marker.open('x',encoding='utf-8') as stream:
            json.dump(dict(output=str(self.root),first_seed=FIRST,last_seed=FIRST+49,
                           frozen_utc=protocol['frozen_utc']),stream,indent=2)
        self.root.mkdir(parents=True)
        write_json(self.root/'manifest.json',protocol)
        self.models[self.root/'manifest.json']=sha256(self.root/'manifest.json')
        records=[]
        for job in protocol['jobs']:
            model=Path(job['model'])
            output=self.root/'evaluations'/(job['name']+'.csv')
            self.job(job['name'],['-m','combat_rl.evaluate','--model',model,'--split','test',
                '--first-seed',FIRST,'--episodes',50,'--output',output])
            record=validate_result(output,job,read_json(model.with_suffix('.json'))['config'],FIRST)
            record.update(csv_sha256=sha256(output),metadata_sha256=sha256(output.with_suffix('.json')),
                          model_sha256=job['model_sha256'],steps=job['steps'])
            records.append(record)
            write_json(self.root/'results.json',summarize(records))
            self.ledger('reserved_test_evaluation',result=record)
        self.check_source()
        self.status('complete',current_job=None,models_unchanged=True,selection_unchanged=True,
                    episodes_completed=350,review_ready=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','survival-root','batch-root','behavior-root','video-deps'):
        parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args()
    worker=Worker(args)
    try:
        worker.run()
    except (Exception,KeyboardInterrupt) as error:
        worker.status('failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
