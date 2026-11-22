"""After the survival queue exits, preserve a five-wave win and an endless demo.

This worker performs inference/recording only. Winning examples are deliberately
selected demonstrations, not extra observations for estimating success rate.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.queue_optimization_windows import clean_supervisor, read_json, sha256, write_json

DEV_SEEDS = list(range(51000, 51050))
TERMINAL_PHASES = {'complete', 'budget_stopped', 'failed'}


def winning_seed_order(root, model):
    """Prefer prior development wins, without reading any held-out results."""
    winners = set()
    for path in sorted((Path(root) / 'evaluations').glob('*.json')):
        meta = read_json(path)
        if (Path(meta.get('model', '')).resolve() != Path(model).resolve()
                or meta.get('split') != 'validation' or meta.get('first_seed') != DEV_SEEDS[0]
                or meta.get('episodes') != len(DEV_SEEDS) or not meta.get('deterministic')):
            continue
        with path.with_suffix('.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if [int(r['scenario_seed']) for r in rows] != DEV_SEEDS:
            raise RuntimeError('Unexpected development scenario list')
        for row in rows:
            if row['success'] == 'True':
                if row['waves_completed'] != '5' or row['terminal_reason'] != 'success':
                    raise RuntimeError('Inconsistent five-wave result')
                winners.add(int(row['scenario_seed']))
    return sorted(winners) + [seed for seed in DEV_SEEDS if seed not in winners]


def verify_video(directory, model_hash, mode='waves', require_win=False):
    directory = Path(directory)
    if (directory / 'INCOMPLETE.txt').exists():
        raise RuntimeError('Incomplete recording')
    record = read_json(directory / 'episode.json')
    metrics = record['metrics']
    if (record['model_sha256'] != model_hash or record.get('mode', 'waves') != mode
            or record['scenario_seed'] not in DEV_SEEDS or not record['deterministic']):
        raise RuntimeError('Recording identity mismatch')
    if require_win and not (metrics['success'] is True and metrics['waves_completed'] == 5
                            and metrics['terminal_reason'] == 'success' and metrics['kills'] == 15):
        raise RuntimeError('This recording is not a complete five-wave win')
    if mode == 'endless' and (metrics.get('task') != 'endless_demo' or metrics['success']
                              or metrics['waves_completed'] != 0):
        raise RuntimeError('Endless results cannot be five-wave successes')
    import imageio_ffmpeg
    for filename, key in [('episode.mp4', 'raw_frames'), ('episode-annotated.mp4', 'annotated_frames')]:
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(directory / filename))
        if frames != record[key] or abs(seconds-frames/20) > 0.06:
            raise RuntimeError('Video frames/duration do not match metadata')
    if record['raw_frames'] != round(metrics['elapsed_seconds']*20)+1:
        raise RuntimeError('Video does not include every simulated tick')
    return record


class Worker:
    def __init__(self, args):
        self.args = args
        self.source = Path(__file__).resolve().parents[1]
        self.root = args.output.resolve()
        self.predecessor = args.predecessor.resolve()
        self.revision = self.git('rev-parse', 'HEAD')
        self.manifest_hash = sha256(self.predecessor / 'manifest.json')
        try:
            self.predecessor_identity = psutil.Process(args.predecessor_pid).create_time()
        except psutil.NoSuchProcess:
            self.predecessor_identity = None
        self.state = {}
        self.active_start = None
        self.model = self.model_hash = None
        self.attempts = []
        sys.path.insert(0, str(args.video_deps.resolve()))

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.source, text=True).strip()

    def source_check(self):
        if self.git('rev-parse', 'HEAD') != self.revision or self.git('status', '--porcelain'):
            raise RuntimeError('Demonstration source changed')
        if sha256(self.predecessor / 'manifest.json') != self.manifest_hash:
            raise RuntimeError('Predecessor manifest changed')
        if self.model and sha256(self.model) != self.model_hash:
            raise RuntimeError('Selected policy changed during inference')

    def predecessor_alive(self):
        if self.predecessor_identity is None:
            return False
        try:
            process = psutil.Process(self.args.predecessor_pid)
            return process.is_running() and process.create_time() == self.predecessor_identity
        except psutil.NoSuchProcess:
            return False

    def status(self, phase, **extra):
        self.state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(self.root / 'status.json', self.state)

    def ledger(self, event, **extra):
        with (self.root / 'ledger.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=event, **extra))+'\n')

    def wait_for_predecessor(self):
        deadline = time.monotonic() + self.args.wait_hours*3600
        self.status('waiting_for_survival', predecessor_pid=self.args.predecessor_pid,
                    note='No simulator or policy update starts while the survival worker is alive.')
        while True:
            self.source_check()
            state = read_json(self.predecessor / 'status.json')
            alive = self.predecessor_alive()
            if not alive:
                if state['phase'] not in TERMINAL_PHASES:
                    raise RuntimeError('Survival worker exited without a terminal status')
                # Every started job must have completed supervisor cleanup.
                for line in (self.predecessor / 'ledger.jsonl').read_text().splitlines():
                    event = json.loads(line)
                    if event['event'] == 'job_started':
                        clean_supervisor(self.predecessor / 'logs' / (event['label']+'.status.json'))
                return state
            if time.monotonic() >= deadline or (self.root / 'STOP_AFTER_CURRENT').exists():
                raise RuntimeError('Waiting stopped; no demonstration simulator started')
            self.status('waiting_for_survival', predecessor_phase=state['phase'],
                        predecessor_job=state.get('current_job'))
            time.sleep(15)

    def remaining(self):
        return self.args.active_hours*3600 - (time.monotonic()-self.active_start)

    def job(self, label, arguments, timeout):
        self.source_check()
        if self.predecessor_alive():
            raise RuntimeError('Predecessor is still running')
        if (self.root / 'STOP_AFTER_CURRENT').exists():
            raise RuntimeError('STOP_AFTER_CURRENT requested')
        # Check shared runtime use, including an instance still starting up.
        for process in psutil.process_iter(['cmdline']):
            try:
                if 'craftground_runtime_mc121' in ' '.join(process.info['cmdline'] or []):
                    raise RuntimeError('Another Minecraft runtime is active')
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(('127.0.0.1', 8000)) == 0:
                raise RuntimeError('Minecraft port is occupied')
        if self.remaining() < timeout+30:
            raise RuntimeError('Not enough demonstration budget for a complete job')
        log = self.root / 'logs' / (label+'.log')
        if log.exists():
            raise RuntimeError('Refusing to reuse a recording job')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.args.video_deps.resolve())
        command = [sys.executable, str(self.source/'scripts/windows_run.py'), '--timeout', str(timeout),
                   '--log', str(log), '--', *map(str, arguments)]
        self.status('recording', current_job=label)
        self.ledger('job_started', label=label, command=command)
        process = subprocess.Popen(command, cwd=self.source, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        while process.poll() is None:
            self.status('recording', remaining_active_seconds=max(0, self.remaining()))
            time.sleep(5)
        record = clean_supervisor(log.with_suffix('.status.json'))
        if process.returncode:
            raise RuntimeError('Recording supervisor failed')
        self.source_check()
        self.ledger('job_completed', label=label, supervisor=record)

    def publish_win(self, directory, record):
        verify_video(directory, self.model_hash, require_win=True)
        for name, source in [('five-wave-win.mp4', 'episode-annotated.mp4'),
                             ('five-wave-win-raw.mp4', 'episode.mp4')]:
            dest = self.root / name
            if dest.exists():
                raise RuntimeError('Demonstration already published')
            shutil.copy2(Path(directory)/source, dest)
            if sha256(dest) != sha256(Path(directory)/source):
                raise RuntimeError('Video copy checksum mismatch')
        write_json(self.root/'five-wave-win.json', dict(record=record, original_directory=str(directory),
            video=str(self.root/'five-wave-win.mp4'), selection='First verified win among existing fixed recordings, then ordered development attempts. Deliberately selected demonstration; do not estimate win rate from these attempts.',
            attempts=self.attempts, model_sha256=self.model_hash))
        self.ledger('five_wave_win_ready', directory=str(directory), metrics=record['metrics'])

    def capture_win(self):
        for path in sorted((self.predecessor/'videos').glob('*/episode.json')):
            record = read_json(path)
            if record['model_sha256'] != self.model_hash:
                continue
            record = verify_video(path.parent, self.model_hash)
            self.attempts.append(dict(existing=True, directory=str(path.parent), metrics=record['metrics'],
                                      scenario_seed=record['scenario_seed']))
            if record['metrics']['success']:
                self.publish_win(path.parent, record)
                return True
        for index, seed in enumerate(winning_seed_order(self.predecessor, self.model)[:self.args.max_attempts], 1):
            # Preserve twenty-five minutes for the endless demo and cleanup.
            if self.remaining() < 900+1500:
                break
            output = self.root/'attempts'/f'win-{index:02d}-scenario{seed}'
            self.job(f'win-{index:02d}', ['-m', 'combat_rl.record_episode', '--model', self.model,
                     '--output-dir', output, '--scenario-seed', seed], timeout=900)
            record = verify_video(output, self.model_hash)
            self.attempts.append(dict(existing=False, directory=str(output), scenario_seed=seed, metrics=record['metrics']))
            write_json(self.root/'win-attempts.json', self.attempts)
            self.ledger('win_attempt', **self.attempts[-1])
            if record['metrics']['success']:
                self.publish_win(output, record)
                return True
        self.ledger('no_recorded_win', attempts=len(self.attempts), note='All attempts preserved; no fabricated success or task changes.')
        return False

    def run(self):
        self.source_check()
        if self.root.exists():
            raise RuntimeError('Use a new demonstration output root')
        self.root.mkdir(parents=True)
        write_json(self.root/'manifest.json', dict(source_revision=self.revision, source=str(self.source),
            predecessor=str(self.predecessor), predecessor_manifest_sha256=self.manifest_hash,
            predecessor_pid=self.args.predecessor_pid, predecessor_create_time=self.predecessor_identity,
            frozen_utc=datetime.now(timezone.utc).isoformat(), wait_hours=self.args.wait_hours,
            active_hours=self.args.active_hours, max_new_win_attempts=self.args.max_attempts,
            endless=dict(scenario_seed=51000, spawn_seconds=2.0, max_seconds=600, observation_wave=5),
            policy_updates=False, holdout_used=False, review_required_before_tuning=True))
        try:
            predecessor_status = self.wait_for_predecessor()
            selected = read_json(self.predecessor/'selection.json')
            if not selected.get('best_seed0'):
                raise RuntimeError('No selected demonstration checkpoint')
            self.model = Path(selected['best_seed0']['model']).resolve()
            self.model_hash = sha256(self.model)
            write_json(self.root/'selected-policy.json', dict(model=str(self.model), sha256=self.model_hash,
                metadata_sha256=sha256(self.model.with_suffix('.json')), selection=selected,
                predecessor_status=predecessor_status,
                note='Use the best development checkpoint for demonstration; preserve its cross-seed confirmation status. No selection uses the new final test.'))
            self.active_start = time.monotonic()
            captured = self.capture_win()
            output = self.root/'endless-scenario51000'
            self.job('endless-demo', ['-m', 'combat_rl.record_episode', '--model', self.model,
                     '--output-dir', output, '--scenario-seed', 51000, '--mode', 'endless',
                     '--spawn-seconds', 2, '--max-seconds', 600], timeout=1200)
            endless = verify_video(output, self.model_hash, mode='endless')
            self.ledger('endless_demo_ready', directory=str(output), metrics=endless['metrics'])
            write_json(self.root/'results.json', dict(five_wave_win_captured=captured,
                win_video=str(self.root/'five-wave-win.mp4') if captured else None,
                endless_video=str(output/'episode-annotated.mp4'), endless_metrics=endless['metrics'],
                model_sha256=self.model_hash, policy_updated=False, next_action='User reviews footage before any calibration or fine-tuning.'))
            self.status('ready_for_review', current_job=None, five_wave_win_captured=captured,
                        endless_kills=endless['metrics']['kills'], policy_updated=False)
        except Exception as error:
            self.status('failed', error=repr(error), policy_updated=False)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor', required=True, type=Path)
    parser.add_argument('--predecessor-pid', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--video-deps', required=True, type=Path)
    parser.add_argument('--wait-hours', type=float, default=20)
    parser.add_argument('--active-hours', type=float, default=1.5)
    parser.add_argument('--max-attempts', type=int, default=30)
    args = parser.parse_args()
    if not (0 < args.wait_hours <= 24 and 1 <= args.active_hours <= 2 and 1 <= args.max_attempts <= 50):
        parser.error('Require wait <=24h, active 1-2h, and 1-50 attempts')
    Worker(args).run()


if __name__ == '__main__':
    main()
