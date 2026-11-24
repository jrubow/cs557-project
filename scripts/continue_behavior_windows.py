"""Resume passive analysis after a supervised failure, reusing verified results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.queue_behavior_windows import Worker as BaseWorker
from scripts.finish_endless_windows import require_idle
from scripts.queue_optimization_windows import read_json,write_json,sha256,clean_supervisor


class Worker(BaseWorker):
    def __init__(self,args):
        self.args,self.source=args,Path(__file__).resolve().parents[1]
        self.root,self.predecessor=args.output.resolve(),args.predecessor.resolve()
        self.previous=args.previous.resolve()
        self.predecessor_identity=None  # Completed batch controller has already exited.
        self.revision=self.git('rev-parse','HEAD')
        self.predecessor_hash=sha256(self.predecessor/'manifest.json')
        self.models,self.specs,self.cached={},{},{}
        self.state=dict(jobs_completed=0,jobs_reused=0,overall_deadline=None)
        self.history_path=self.predecessor/'critical-history.json'
        self.history_hash=sha256(self.history_path)
        self.history=read_json(self.history_path)
        for record in self.history['checkpoints']:
            if sha256(self.remember(record['model']))!=record['sha256']:
                raise RuntimeError('Historical checkpoint changed')
        self.check_source()

    def predecessor_alive(self):
        return False  # wait() requires all old controller identities to be absent first.

    def wait(self):
        require_idle(self.args.old_pids)
        old=self.previous
        state,manifest=read_json(old/'status.json'),read_json(old/'manifest.json')
        if state['phase']!='failed' or manifest['predecessor']!=str(self.predecessor):
            raise RuntimeError('Expected failed analysis of the same completed batch')
        events=[json.loads(line) for line in (old/'ledger.jsonl').read_text().splitlines()]
        started={e['data']['label']:e['data'] for e in events if e['event']=='job_started'}
        completed={e['data']['label']:e['data'] for e in events if e['event']=='job_completed'}
        self.cached={e['data']['result']['name']:e['data'] for e in events if e['event']=='behavior_evaluation'}
        if set(completed)!=set(self.cached) or state['jobs_completed']!=len(completed):
            raise RuntimeError('Incomplete prior result ledger')
        failures=[]
        for name,event in started.items():
            path=old/'logs'/(name+'.status.json')
            receipt=read_json(path)
            command=event['command']
            if receipt['command'][1:]!=command[command.index('--')+1:] or receipt['remaining_descendant_pids']:
                raise RuntimeError('Prior job identity/cleanup mismatch')
            if name in completed:
                if clean_supervisor(path)!=completed[name]['supervisor']:
                    raise RuntimeError('Prior successful receipt changed')
            elif receipt['exit_code']!=0:
                failures.append(dict(name=name,receipt=receipt))
            else:
                raise RuntimeError('Unrecorded completed job requires explicit audit')
        if len(failures)!=1 or failures[0]['name']!=state['current_job']:
            raise RuntimeError('Expected one preserved failed evaluation')
        self.old_revision=manifest['source_revision']
        evidence={str(old/name):sha256(old/name) for name in ('manifest.json','status.json','ledger.jsonl')}
        self.models.update({Path(path):digest for path,digest in evidence.items()})
        write_json(self.root/'recovery.json',dict(previous=str(old),source_revision=self.old_revision,
            reusable_evaluations=len(self.cached),evidence_sha256=evidence,failed_attempts=failures,
            note='Original failure/partial outputs retained. Same scenario requests; no result selection or training.'))
        super().wait()  # Verify the completed batch and all its supervisors too.

    def evaluate_behavior(self,name,model,spec,first=53000,episodes=10,policy='ppo'):
        if name not in self.cached:
            return super().evaluate_behavior(name,model,spec,first,episodes,policy)
        saved=self.cached[name]
        directory=Path(saved['result']['output'])
        if sha256(directory/'metadata.json')!=saved['metadata_sha256']:
            raise RuntimeError('Cached behavior metadata changed')
        result=self.read_behavior(name,directory,model,spec,first,episodes,policy,self.old_revision)
        if result!=saved['result']:
            raise RuntimeError('Cached behavior evidence changed')
        self.state['jobs_reused']+=1
        self.ledger('behavior_reused',name=name,output=str(directory),source_revision=self.old_revision)
        self.status('reusing_completed_analysis',current_job=name)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','predecessor','previous','video-deps'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--predecessor-pid',required=True,type=int)
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
