"""After batch optimization, compare passive combat behaviors across checkpoints."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from combat_rl.behavior import DEFINITIONS, aggregate
from scripts.queue_endless_windows import Worker as BaseWorker
from scripts.queue_optimization_windows import clean_supervisor,read_json,write_json,sha256


class Worker(BaseWorker):
    def __init__(self,args):
        args.predecessor_control = args.predecessor
        super().__init__(args)
        self.history_path = self.predecessor/'critical-history.json'
        self.history_hash = sha256(self.history_path)
        self.history = read_json(self.history_path)
        for record in self.history['checkpoints']:
            model = self.remember(record['model'])
            if sha256(model)!=record['sha256']:
                raise RuntimeError('Historical checkpoint changed')

    def wait(self):
        while True:
            self.check_source()
            if sha256(self.history_path)!=self.history_hash:
                raise RuntimeError('Historical checkpoint manifest changed')
            if (self.root/'STOP_AFTER_CURRENT').exists():
                raise InterruptedError('STOP_AFTER_CURRENT requested')
            state = read_json(self.predecessor/'status.json')
            if not self.predecessor_alive():
                if state['phase']!='complete':
                    raise RuntimeError('Batch worker exited without successful completion')
                for line in (self.predecessor/'ledger.jsonl').read_text().splitlines():
                    event = json.loads(line)
                    if event['event']=='job_started':
                        details = event['data']
                        status = clean_supervisor(self.predecessor/'logs'/(details['label']+'.status.json'))
                        command = details['command']
                        if status['command'][1:]!=command[command.index('--')+1:]:
                            raise RuntimeError('Predecessor supervisor command mismatch')
                self.status('predecessor_complete',predecessor_phase='complete',predecessor_job=None)
                return
            self.status('waiting_for_batch',predecessor_pid=self.args.predecessor_pid,
                        predecessor_phase=state['phase'],predecessor_job=state.get('current_job'))
            time.sleep(15)

    def evaluate_behavior(self,name,model,spec,first=53000,episodes=10,policy='ppo'):
        model = self.remember(model)
        directory = self.root/'evaluations'/name
        self.job(name,['-m','combat_rl.behavior_eval','--model',model,'--task-spec',self.spec_path(spec),
            '--output',directory,'--first-seed',first,'--episodes',episodes,'--policy',policy])
        meta,records = read_json(directory/'metadata.json'),read_json(directory/'episodes.json')
        if (meta['model_sha256']!=sha256(model) or meta['model_metadata_sha256']!=sha256(model.with_suffix('.json'))
                or meta['task']!=spec or meta['policy']!=policy or meta['policy_updated']
                or meta['first_seed']!=first or meta['episodes']!=episodes or meta['split']!='validation'
                or meta['deterministic']!=(policy!='random')
                or meta['config']!=read_json(model.with_suffix('.json'))['config']
                or not 50000<=first<first+episodes<=100000 or meta['definitions']!=DEFINITIONS
                or meta['runtime']['git_revision']!=self.revision or meta['runtime']['git_status']
                or [r['scenario_seed'] for r in records]!=list(range(first,first+episodes))
                or meta['trace_sha256']!=sha256(directory/'ticks.csv.gz')):
            raise RuntimeError('Behavior evaluation provenance mismatch')
        summary = read_json(directory/'summary.json')
        recomputed = aggregate(records)
        if any(summary[key]!=value for key,value in recomputed.items()):
            raise RuntimeError('Behavior summary does not match episode evidence')
        result = dict(name=name,model=str(model),steps=meta['model_steps'],policy=policy,task=spec,
                      output=str(directory),summary=summary,model_sha256=sha256(model))
        self.ledger('behavior_evaluation',result=result,metadata_sha256=sha256(directory/'metadata.json'))
        return result

    def record_behavior(self,name,model,spec,seed):
        directory = self.root/'videos'/name
        mode = 'waves' if spec['mode']=='waves' else 'batch'
        args = ['-m','combat_rl.record_episode','--model',model,'--output-dir',directory,
                '--scenario-seed',seed,'--mode',mode,'--track-behavior']
        if mode=='batch':
            args += ['--batch-size',spec['batch_size'],'--max-seconds',spec['max_seconds']]
        self.job('record-'+name,args)
        report = read_json(directory/'episode.json')
        if ((directory/'INCOMPLETE.txt').exists() or report['model_sha256']!=sha256(model)
                or report['scenario_seed']!=seed or report['policy_updated'] or report['mode']!=mode
                or report['behavior_summary']['counts']['ticks']!=round(report['metrics']['elapsed_seconds']*20)):
            raise RuntimeError('Behavior recording provenance mismatch')
        import imageio_ffmpeg
        for filename,key in [('episode.mp4','raw_frames'),('episode-annotated.mp4','annotated_frames')]:
            frames,seconds = imageio_ffmpeg.count_frames_and_secs(str(directory/filename))
            if frames!=report[key] or abs(seconds-frames/20)>.06:
                raise RuntimeError('Behavior video frame mismatch')
        return dict(directory=str(directory),metrics=report['metrics'],summary=report['behavior_summary'])

    def run(self):
        self.root.mkdir(parents=True,exist_ok=False)
        write_json(self.root/'manifest.json',dict(source_revision=self.revision,predecessor=str(self.predecessor),
            predecessor_pid=self.args.predecessor_pid,predecessor_create_time=self.predecessor_identity,
            predecessor_manifest_sha256=self.predecessor_hash,history_sha256=self.history_hash,
            definitions=DEFINITIONS,history_scenarios=[53000,53009],batch_comparison_scenarios=[53100,53119],
            policy_updated=False,held_out_test_used=False,overall_deadline=None))
        self.wait()
        results = read_json(self.predecessor/'results.json')
        selection = read_json(self.predecessor/'selection.json')
        parent,selected = self.remember(selection['parent']),self.remember(results['selected'])
        if sha256(selected)!=results['selected_sha256'] or results['task']!=selection['task']:
            raise RuntimeError('Completed batch selection changed')
        write_json(self.root/'selected-inputs.json',dict(parent=str(parent),parent_sha256=sha256(parent),
            selected=str(selected),selected_sha256=sha256(selected),task=results['task'],
            predecessor_results_sha256=sha256(self.predecessor/'results.json')))
        references = [self.evaluate_behavior(policy,parent,{'mode':'waves'},policy=policy) for policy in ('random','steady')]
        history = []
        for record in self.history['checkpoints']:
            history.append(self.evaluate_behavior('history-'+record['name'],record['model'],{'mode':'waves'}))
            write_json(self.root/'history-progress.json',history)
        comparisons = [self.evaluate_behavior('batch-'+label,model,results['task'],53100,20)
                       for label,model in [('parent',parent),('selected',selected)]]
        comparisons.append(self.evaluate_behavior('waves-selected',selected,{'mode':'waves'}))
        videos = [self.record_behavior('parent-waves',parent,{'mode':'waves'},53000),
                  self.record_behavior('parent-batch',parent,results['task'],53100),
                  self.record_behavior('selected-batch',selected,results['task'],53100)]
        examples = []
        for result in history+comparisons:
            for episode in read_json(Path(result['output'])/'episodes.json'):
                eligible = [w for w in episode['windows'] if not w['partial'] and w['group']=='retreat'
                            and w['damage_dealt']>0 and w['damage_taken']==0]
                if eligible:
                    examples.append(dict(evaluation=result['name'],scenario=episode['scenario_seed'],
                        first_three_windows=eligible[:3],note='Trace evidence; separate videos are fresh rollouts, not replays.'))
        report = dict(history=history,references=references,batch_and_transfer=comparisons,videos=videos,
            example_retreat_windows=examples,definitions=DEFINITIONS,critical_sensor_validated=self.history['sensor_validated'],
            note='Descriptive observations, not causal attribution or proof of intent. Ten episodes/checkpoint; '
                 'different context exposure and initial action bias affect rates. No policy or reward updates.')
        write_json(self.root/'results.json',report)
        plot_history(history,self.root/'behavior-history.png')
        self.status('complete',current_job=None,review_ready=True)


def plot_history(history,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(3,3,figsize=(13,10),layout='constrained')
    plots = [('yaw_correction_fraction','Camera turns reducing yaw error'),
             ('aligned_attack_fraction','Attacks in geometric aiming band'),
             ('damage_per_attack_pulse','Damage per attack request'),
             ('near_wall_fraction','Fraction of ticks near walls'),
             ('critical_cues_per_100_attack_pulses','Critical sound cues / 100 requests'),
             ('retreat','Unanswered retreat windows / minute'),
             ('focus:mean_kill_seconds','Mean first damage to death (completed only, s)'),
             ('focus:switch_before_kill_fraction','Damage-target switches before killing'),
             ('focus:multiple_wounded_fraction','Multiple wounded / multi-enemy ticks')]
    for axis,(key,title) in zip(axes.flat,plots):
        values = []
        for record in history:
            score = record['summary']
            value = (score['focus_fire'][key.split(':')[1]] if key.startswith('focus:') else
                     1200*score['movement_windows']['retreat']['unanswered_damage_windows']/score['counts']['ticks']
                     if key=='retreat' else score[key])
            values.append(float('nan') if value is None else value)
        axis.plot([r['steps']/1e6 for r in history],values,marker='o')
        axis.set(title=title,xlabel='Training decisions (millions)')
        axis.grid(alpha=.25)
    fig.suptitle('C seed 0: observed combat behavior across checkpoints\n10 matched development scenarios; descriptive proxies, not proof of intent')
    fig.supxlabel('Kill-time means exclude unfinished, initially injured and observation-gap cases; see counts in summaries.')
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('output','predecessor','video-deps'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--predecessor-pid',required=True,type=int)
    args = parser.parse_args()
    sys.path.insert(0,str(args.video_deps.resolve()))
    worker = Worker(args)
    try:
        worker.run()
    except (Exception,KeyboardInterrupt) as error:
        worker.status('stopped' if isinstance(error,InterruptedError) else 'failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
