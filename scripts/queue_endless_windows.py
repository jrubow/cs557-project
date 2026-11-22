"""Sequential, validation-only batch-survival calibration and critical-cue study.

Wait for the exact predecessor controller to exit successfully. There is no
overall time deadline. Finite step targets, per-job supervision, immutable
source/checkpoints and STOP_AFTER_CURRENT protect the shared Minecraft runtime.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import zipfile

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.queue_optimization_windows import clean_supervisor, read_json, sha256, write_json

PROFILES = {**{f'batch{k}': dict(mode='batch', batch_size=k, max_seconds=300) for k in (1, 2, 3, 5)},
            **{f'timed{s}': dict(mode='timed', spawn_seconds=s, max_seconds=300) for s in (2, 4, 6)}}
TARGETS = (250_000, 500_000, 1_000_000, 2_000_000)


def select_batch(scores):
    batches = [name for name in scores if name.startswith('batch')]
    viable = [name for name in batches if scores[name]['median_seconds'] >= 60
              and scores[name]['median_kills'] >= 10]
    if viable:
        return max(viable, key=lambda name: PROFILES[name]['batch_size'])
    return max(batches, key=lambda name: (scores[name]['median_kills'], scores[name]['median_seconds'],
                                         PROFILES[name]['batch_size']))


def improves(candidate, baseline):
    return (candidate['mean_kills'] >= baseline['mean_kills']+max(2, baseline['mean_kills']*.1)
            and candidate['mean_seconds'] >= baseline['mean_seconds'])


def summarize(path, model, spec, first, episodes, revision):
    meta = read_json(Path(path).with_suffix('.json'))
    if (meta['model_sha256'] != sha256(model) or meta['evaluation_task'] != spec
            or meta['split'] != 'validation' or meta['first_seed'] != first or meta['episodes'] != episodes
            or not meta['deterministic'] or meta['policy_updated']
            or meta['runtime']['git_revision'] != revision or meta['runtime']['git_status']
            or meta['config'] != read_json(Path(model).with_suffix('.json'))['config']
            or Path(meta['model']).resolve() != Path(model).resolve()
            or not 50000 <= first < first+episodes <= 100000):
        raise RuntimeError('Evaluation provenance or split mismatch')
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if [int(row['scenario_seed']) for row in rows] != list(range(first, first+episodes)):
        raise RuntimeError('Incomplete or duplicated evaluation scenarios')
    fields = ('kills', 'elapsed_seconds', 'critical_sound_events', 'attack_pulses', 'game_ticks',
              'damage_dealt', 'damage_taken', 'waves_completed', 'batches_completed')
    for row in rows:
        if any(not math.isfinite(float(row[key])) or float(row[key]) < 0 for key in fields):
            raise RuntimeError('Invalid evaluation metrics')
        if row['mode'] != spec['mode'] or Path(row['model']).resolve() != Path(model).resolve():
            raise RuntimeError('Episode model/task mismatch')
        if spec['mode'] != 'waves' and (row['success'] != 'False' or float(row['waves_completed']) != 0):
            raise RuntimeError('Survival task cannot count as five-wave success')
    kills = [float(r['kills']) for r in rows]
    seconds = [float(r['elapsed_seconds']) for r in rows]
    cues = sum(int(r['critical_sound_events']) for r in rows)
    pulses = sum(int(r['attack_pulses']) for r in rows)
    return dict(episodes=episodes, mean_kills=statistics.mean(kills), median_kills=statistics.median(kills),
        mean_seconds=statistics.mean(seconds), median_seconds=statistics.median(seconds),
        capped_episodes=sum(r['terminal_reason'] == 'recording_time_limit' for r in rows),
        five_wave_success_rate=sum(r['success'] == 'True' for r in rows)/episodes,
        mean_waves=statistics.mean(float(r['waves_completed']) for r in rows),
        critical_sound_events=cues, episodes_with_critical_cues=sum(int(r['critical_sound_events']) > 0 for r in rows),
        critical_cues_per_100_attack_pulses=100*cues/pulses if pulses else None,
        critical_cues_per_game_minute=60*cues/sum(seconds) if sum(seconds) else None,
        csv_sha256=sha256(path))


class Worker:
    def __init__(self, args):
        self.args = args
        self.source = Path(__file__).resolve().parents[1]
        self.root = args.output.resolve()
        if self.root.exists():
            raise RuntimeError('Use a fresh output directory; existing status is preserved')
        self.predecessor = args.predecessor.resolve()
        self.control = args.predecessor_control.resolve()
        self.revision = self.git('rev-parse', 'HEAD')
        self.predecessor_hash = sha256(self.predecessor/'manifest.json')
        self.predecessor_identity = psutil.Process(args.predecessor_pid).create_time()
        self.models, self.specs = {}, {}
        self.state = dict(jobs_completed=0, overall_deadline=None)
        self.check_source()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.source, text=True).strip()

    def check_source(self):
        if self.git('rev-parse', 'HEAD') != self.revision or self.git('status', '--porcelain'):
            raise RuntimeError('Queue source changed')
        if sha256(self.predecessor/'manifest.json') != self.predecessor_hash:
            raise RuntimeError('Predecessor manifest changed')
        for path, expected in self.models.items():
            if sha256(path) != expected:
                raise RuntimeError(f'Checkpoint or metadata changed: {path}')

    def remember(self, model):
        model = Path(model).resolve()
        self.models[model] = sha256(model)
        self.models[model.with_suffix('.json')] = sha256(model.with_suffix('.json'))
        return model

    def status(self, phase, **extra):
        self.state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(self.root/'status.json', self.state)

    def ledger(self, event, **data):
        with (self.root/'ledger.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=event, data=data))+'\n')

    def predecessor_alive(self):
        try:
            process = psutil.Process(self.args.predecessor_pid)
            return process.is_running() and process.create_time() == self.predecessor_identity
        except psutil.NoSuchProcess:
            return False

    def wait(self):
        while True:
            self.check_source()
            if (self.root/'STOP_AFTER_CURRENT').exists():
                raise InterruptedError('STOP_AFTER_CURRENT requested')
            state, control = read_json(self.predecessor/'status.json'), read_json(self.control/'status.json')
            if not self.predecessor_alive():
                if state['phase'] != 'complete' or control['phase'] != 'complete':
                    raise RuntimeError('Predecessor exited without both successful completion states')
                for line in (self.predecessor/'ledger.jsonl').read_text().splitlines():
                    event = json.loads(line)
                    if event['event'] == 'job_started':
                        clean_supervisor(self.predecessor/'logs'/(event['label']+'.status.json'))
                return
            self.status('waiting_for_survival', predecessor_pid=self.args.predecessor_pid,
                        predecessor_phase=state['phase'], predecessor_job=state.get('current_job'))
            time.sleep(15)

    def job(self, label, arguments, timeout=5400):
        self.check_source()
        if self.predecessor_alive():
            raise RuntimeError('Predecessor still running')
        if (self.root/'STOP_AFTER_CURRENT').exists():
            raise InterruptedError('STOP_AFTER_CURRENT requested')
        for process in psutil.process_iter(['cmdline']):
            try:
                if 'craftground_runtime_mc121' in ' '.join(process.info['cmdline'] or []):
                    raise RuntimeError('Another Minecraft runtime is active')
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(('127.0.0.1', 8000)) == 0:
                raise RuntimeError('Minecraft port occupied')
        log = self.root/'logs'/(label+'.log')
        log.parent.mkdir(parents=True,exist_ok=True)
        if log.exists():
            raise RuntimeError('Refusing to overwrite a supervised job')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.args.video_deps.resolve())
        command = [sys.executable, str(self.source/'scripts/windows_run.py'), '--timeout', str(timeout),
                   '--log', str(log), '--', *map(str, arguments)]
        self.status('running', current_job=label)
        self.ledger('job_started', label=label, command=command)
        with log.with_suffix('.supervisor.log').open('x',encoding='utf-8') as supervisor_log:
            process = subprocess.Popen(command, cwd=self.source, env=env, stdout=supervisor_log,
                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            while process.poll() is None:
                self.status('running')
                time.sleep(5)
        record = clean_supervisor(log.with_suffix('.status.json'))
        if process.returncode:
            raise RuntimeError('Supervisor failed')
        self.check_source()
        self.state['jobs_completed'] += 1
        self.ledger('job_completed', label=label, supervisor=record)

    def spec_path(self, spec):
        key = json.dumps(spec, sort_keys=True)
        if key not in self.specs:
            path = self.root/'task-specs'/f'{len(self.specs):02d}.json'
            write_json(path, spec)
            self.specs[key] = path
        path = self.specs[key]
        if read_json(path) != spec:
            raise RuntimeError('Frozen task spec changed')
        return path

    def evaluate(self, name, model, spec, first, episodes):
        path = self.root/'evaluations'/(name+'.csv')
        self.job(name, ['-m','combat_rl.survival_task','--operation','evaluate','--model',model,
            '--task-spec', self.spec_path(spec), '--output',path,'--first-seed',first,'--episodes',episodes])
        score = summarize(path, model, spec, first, episodes, self.revision)
        self.ledger('evaluation', name=name, model=str(model), task=spec, score=score)
        return score

    def history(self, parent):
        self.job('critical-sensor-controls', ['-m','combat_rl.crit_probe','--model',parent,
                                            '--output',self.root/'critical-controls.json'])
        control = read_json(self.root/'critical-controls.json')
        available = list((self.args.original_root/'C_seed0').glob('step_*.zip'))
        checkpoints = [(f'c{target}', min(available, key=lambda p: (abs(int(p.stem[5:])-target), p.name)))
                       for target in (10000, 50000, 100000, 300000, 500000)]
        checkpoints += [('c1000000', self.args.original_root/'C_seed0/final.zip'),
                        ('c2000000', self.args.optimization_root/'models/more_2000000_seed0/final.zip')]
        checkpoints += [(f'survival10_{t}', self.predecessor/f'models/survival10_{t}_seed0/final.zip')
                        for t in (3000000,4000000,5000000)]
        rows = []
        for name, path in checkpoints:
            model = self.remember(path)
            score = self.evaluate('critical-history-'+name, model, {'mode':'waves'}, 53000, 10)
            rows.append(dict(name=name, model=str(model), sha256=sha256(model),
                             steps=read_json(model.with_suffix('.json'))['timesteps'], score=score))
        first = next((r['steps'] for r in rows if r['score']['critical_sound_events']), None)
        write_json(self.root/'critical-history.json', dict(sensor_validated=control['validated'], checkpoints=rows,
            earliest_observed_checkpoint_steps=first if control['validated'] else None,
            note='Matched fresh development rollouts, ten scenarios per checkpoint. Sound cues are not a server hit counter. '
                 'Earliest sampled observation is not the exact learning step or proof of causation. '
                 'Zeros with failed sensor controls are inconclusive. Policy receives neither sound nor a crit bonus.'))

    def record(self, name, model, spec, seed):
        directory = self.root/'videos'/name
        mode = 'batch' if spec['mode'] == 'batch' else 'endless'
        args = ['-m','combat_rl.record_episode','--model',model,'--output-dir',directory,
                '--mode',mode,'--scenario-seed',seed,'--max-seconds',spec['max_seconds'],'--track-crits']
        args += ['--batch-size',spec['batch_size']] if mode == 'batch' else ['--spawn-seconds',spec['spawn_seconds']]
        self.job('record-'+name, args)
        record = read_json(directory/'episode.json')
        if ((directory/'INCOMPLETE.txt').exists() or record['model_sha256'] != sha256(model)
                or record['scenario_seed'] != seed or record['mode'] != mode or record['policy_updated']
                or record['raw_frames'] != round(record['metrics']['elapsed_seconds']*20)+1):
            raise RuntimeError('Recording provenance/frame mismatch')
        import imageio_ffmpeg
        for filename, key in [('episode.mp4','raw_frames'),('episode-annotated.mp4','annotated_frames')]:
            frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(directory/filename))
            if frames != record[key] or abs(seconds-frames/20) > .06:
                raise RuntimeError('Encoded recording failed verification')
        return dict(directory=str(directory), metrics=record['metrics'], telemetry=record['combat_telemetry'])

    def run(self):
        if self.root.exists():
            raise RuntimeError('Use a fresh output directory')
        self.root.mkdir(parents=True)
        write_json(self.root/'manifest.json', dict(source_revision=self.revision, profiles=PROFILES,
            parent_selection='Final predecessor development-selected best_seed0; no test selection',
            predecessor=str(self.predecessor), predecessor_control=str(self.control),
            predecessor_pid=self.args.predecessor_pid, predecessor_create_time=self.predecessor_identity,
            predecessor_manifest_sha256=self.predecessor_hash, additional_step_targets=TARGETS,
            calibration_seeds=[52000,52009], development_seeds=[52100,52119], confirmation_seeds=[52200,52219],
            critical_history_seeds=[53000,53009], overall_time_limit=None, training_seeds=[0],
            protocol=str(self.source/'docs/endless-calibration.md'), held_out_test_used=False))
        self.wait()
        parent = self.remember(read_json(self.predecessor/'selection.json')['best_seed0']['model'])
        initial_steps = read_json(parent.with_suffix('.json'))['timesteps']
        self.status('calibrating', parent=str(parent), parent_sha256=sha256(parent))
        scores = {name:self.evaluate('calibration-'+name, parent, spec, 52000, 10)
                  for name, spec in PROFILES.items()}
        chosen = select_batch(scores)
        spec = PROFILES[chosen]
        write_json(self.root/'calibration.json', dict(profiles=scores, selected=chosen, frozen_task=spec))
        baseline = self.evaluate('baseline-batch', parent, spec, 52100, 20)
        self.evaluate('baseline-timed2', parent, PROFILES['timed2'], 52100, 20)
        self.history(parent)
        best, best_score, current, candidates = parent, baseline, parent, []
        for target in TARGETS:
            previous = read_json(current.with_suffix('.json'))
            additional = initial_steps+target-previous['timesteps']
            directory = self.root/'models'/f'additional_{target}'
            self.job(f'train-additional-{target}', ['-m','combat_rl.survival_task','--operation','train',
                '--model',current,'--task-spec',self.spec_path(spec),'--output',directory,
                '--additional-steps',additional], timeout=21600)
            model = directory/'final.zip'
            meta = read_json(model.with_suffix('.json'))
            rollout = meta['config']['training']['n_steps']
            with zipfile.ZipFile(model) as archive:
                actual = json.loads(archive.read('data'))['num_timesteps']
            if (meta['task_spec'] != spec or meta['resume_sha256'] != sha256(current)
                    or meta['task_start_timesteps'] != initial_steps or actual != meta['timesteps']
                    or not initial_steps+target <= actual < initial_steps+target+rollout
                    or meta['runtime']['git_revision'] != self.revision or meta['runtime']['git_status']):
                raise RuntimeError('Transfer checkpoint provenance mismatch')
            current = self.remember(model)
            score = self.evaluate(f'candidate-{target}', current, spec, 52100, 20)
            candidates.append(dict(model=str(current), sha256=sha256(current), additional_target=target, score=score))
            if improves(score, baseline) and (score['mean_kills'], score['mean_seconds']) > (best_score['mean_kills'], best_score['mean_seconds']):
                best, best_score = current, score
            write_json(self.root/'selection.json', dict(parent=str(parent), selected=str(best), selected_sha256=sha256(best),
                task=spec, baseline=baseline, selected_score=best_score, candidates=candidates, held_out_test_used=False))
            if target >= 500000 and score['mean_kills'] < .8*baseline['mean_kills'] and score['mean_seconds'] <= baseline['mean_seconds']:
                self.ledger('regression_stop', target=target, reason='Kills below 80% of baseline with no survival gain')
                break
        # Selection is locked before fresh confirmation; these cannot choose a different checkpoint.
        confirmation = {name:self.evaluate('confirmation-'+name, model, spec, 52200, 20)
                        for name, model in [('parent',parent),('selected',best)]}
        transfer = {name:self.evaluate('five-wave-transfer-'+name, model, {'mode':'waves'}, 51000, 50)
                    for name, model in [('parent',parent),('selected',best)]}
        timed = self.evaluate('selected-timed2', best, PROFILES['timed2'], 52100, 20)
        videos = [self.record('parent-batch-52000', parent, spec, 52000)]
        videos += [self.record(f'selected-batch-{seed}', best, spec, seed) for seed in range(52000,52003)]
        videos += [self.record('selected-timed2-52000', best, PROFILES['timed2'], 52000)]
        write_json(self.root/'results.json', dict(selected=str(best), selected_sha256=sha256(best), task=spec,
            calibration=scores, baseline=baseline, selected_development=best_score, confirmation=confirmation,
            improvement_confirmed=improves(confirmation['selected'], confirmation['parent']),
            five_wave_transfer=transfer, timed_transfer=timed, videos=videos,
            note='Exploratory single-training-seed task transfer. No original or reserved test scenarios used. '
                 'Episode caps are right-censored survival times; recordings use fixed seeds, without winner selection.'))
        self.status('complete', current_job=None, selected=str(best), review_ready=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('output','predecessor','predecessor-control','original-root','optimization-root','video-deps'):
        parser.add_argument('--'+key, required=True, type=Path)
    parser.add_argument('--predecessor-pid', required=True, type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(args.video_deps.resolve()))
    worker = Worker(args)
    try:
        worker.run()
    except (Exception, KeyboardInterrupt) as error:
        worker.status('stopped' if isinstance(error, InterruptedError) else 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
