"""Finish an existing survival protocol without an overall wall-clock deadline.

Wait for a graceful between-job handoff, validate/reuse completed jobs, and keep
the original frozen training source. No running training process is interrupted.
The original manifest stays immutable; the user's amendment is a separate record.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import queue_survival_windows as q


class ContinuationWorker(q.Worker):
    def __init__(self, args):
        super().__init__(args)
        self.controller_source = self.source
        self.controller_revision = self.revision
        self.source = args.training_source.resolve()
        self.revision = self.git('rev-parse', 'HEAD')
        self.control = args.control.resolve()
        self.original_manifest_hash = q.sha256(self.root/'manifest.json')
        try:
            self.predecessor_identity = psutil.Process(args.predecessor_pid).create_time()
        except psutil.NoSuchProcess:
            self.predecessor_identity = None
        self.completed_events = []
        self.ready = False

    def controller_git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.controller_source, text=True).strip()

    def source_check(self):
        super().source_check()
        if (self.controller_git('rev-parse', 'HEAD') != self.controller_revision
                or self.controller_git('status', '--porcelain')):
            raise RuntimeError('Continuation controller source changed')
        if q.sha256(self.root/'manifest.json') != self.original_manifest_hash:
            raise RuntimeError('Original manifest changed')

    def remaining(self):
        return float('inf')

    def status(self, phase, **extra):
        self.state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-self.started, remaining_seconds=None,
            overall_time_limit_removed=True, controller_revision=self.controller_revision,
            training_source_revision=self.revision, jobs_completed=self.jobs, **extra)
        q.write_json(self.root/'status.json', self.state)
        q.write_json(self.control/'status.json', self.state)

    def predecessor_alive(self):
        if self.predecessor_identity is None:
            return False
        try:
            process = psutil.Process(self.args.predecessor_pid)
            return process.is_running() and process.create_time() == self.predecessor_identity
        except psutil.NoSuchProcess:
            return False

    def initialize(self):
        self.source_check()
        if self.control.exists():
            raise RuntimeError('Use a new continuation control directory')
        self.control.mkdir(parents=True)
        q.write_json(self.control/'manifest.json', dict(controller_revision=self.controller_revision,
            controller_source=str(self.controller_source), training_source=str(self.source),
            training_revision=self.revision, output=str(self.root),
            original_manifest_sha256=self.original_manifest_hash,
            predecessor_pid=self.args.predecessor_pid, predecessor_create_time=self.predecessor_identity,
            handoff_token=self.args.handoff_token, overall_time_limit=None,
            reason='User requested removal of the hard queue cutoff to avoid restarting completed runs.'))
        while self.predecessor_alive():
            self.source_check()
            q.write_json(self.control/'status.json', dict(phase='waiting_for_current_job',
                updated_utc=datetime.now(timezone.utc).isoformat(), predecessor_pid=self.args.predecessor_pid,
                note='Current supervised training continues unchanged. Handoff occurs only after cleanup.'))
            time.sleep(5)
        state = q.read_json(self.root/'status.json')
        if state['phase'] not in ('complete', 'budget_stopped'):
            raise RuntimeError('Predecessor did not finish a clean between-job handoff')
        manifest = q.read_json(self.root/'manifest.json')
        if (Path(manifest['source']).resolve() != self.source or manifest['source_revision'] != self.revision
                or manifest['development_scenarios'] != list(range(51000, 51050))
                or manifest['reserved_test_scenarios'] != list(range(200000, 200050))
                or manifest['recipes'] != {r: q.recipe_config(self.parents[0]['config'], r) for r in q.RECIPES}):
            raise RuntimeError('Continuation does not match the original scientific protocol')
        for seed, (model, original) in enumerate(zip(self.parent_models, manifest['parents'])):
            if (Path(original['model']).resolve() != model or original['model_sha256'] != q.sha256(model)
                    or original['metadata_sha256'] != q.sha256(model.with_suffix('.json'))
                    or self.parents[seed]['config'] != manifest['parent_config']):
                raise RuntimeError('Parent model identity changed')
        self.completed_events = [json.loads(line) for line in (self.root/'ledger.jsonl').read_text().splitlines()]
        for event in self.completed_events:
            if event['event'] == 'job_started':
                q.clean_supervisor(self.root/'logs'/(event['label']+'.status.json'))
        marker = self.root/'STOP_AFTER_CURRENT'
        marker_record = q.read_json(marker) if marker.exists() else None
        if marker_record and marker_record.get('handoff_token') != self.args.handoff_token:
            raise RuntimeError('An unrelated stop request must not be removed')
        amendment = dict(authorized_change='Remove overall queue deadline; reuse completed work',
            utc=datetime.now(timezone.utc).isoformat(), old_max_hours=manifest['max_hours'], new_max_hours=None,
            original_manifest_sha256=self.original_manifest_hash, training_source_revision=self.revision,
            controller_source_revision=self.controller_revision, predecessor_status=state,
            handoff_marker=marker_record, unchanged='Task, recipes, scenarios, checkpoints and selection rules',
            remaining_protection='Per-job six-hour supervisor timeout and descendant cleanup; STOP_AFTER_CURRENT still works.')
        q.write_json(self.control/'amendment.json', amendment)
        q.write_json(self.root/'time-limit-amendment.json', amendment)
        if marker_record:
            marker.unlink()
        self.ready = True
        self.ledger('controller_handoff', **amendment)
        self.status('resuming_completed_protocol', current_job=None)

    def job(self, label, arguments, estimate, reserve=0):
        status = self.root/'logs'/(label+'.status.json')
        if not status.exists():
            return super().job(label, arguments, estimate, reserve)
        self.source_check()
        record = q.clean_supervisor(status)
        if record['command'][1:] != list(map(str, arguments)):
            raise RuntimeError(f'Completed job arguments changed: {label}')
        if not any(e['event'] == 'job_completed' and e['label'] == label for e in self.completed_events):
            raise RuntimeError('Completed supervisor lacks its original completion ledger')
        for event in self.completed_events:
            if event['event'] == 'evaluation' and event['name'] == label:
                if q.sha256(self.root/'evaluations'/(label+'.csv')) != event['csv_sha256']:
                    raise RuntimeError('Completed evaluation changed')
        self.jobs += 1
        self.ledger('job_reused', label=label, supervisor_sha256=q.sha256(status))
        self.status('reusing_completed_jobs', current_job=label)
        return record['wall_seconds']

    def train(self, recipe, seed, target, parent, first=False, reserve=0):
        name = f'{recipe}_{target}_seed{seed}'
        directory = self.root/'models'/name
        if not directory.exists():
            return super().train(recipe, seed, target, parent, first, reserve)
        prior = q.read_json(Path(parent).with_suffix('.json'))
        config = q.recipe_config(prior['config'], recipe) if first else copy.deepcopy(prior['config'])
        config['training']['total_timesteps'] = target
        frozen = self.root/'configs'/(name+'.json')
        if q.read_json(frozen) != config:
            raise RuntimeError('Saved training settings differ')
        model = directory/'final.zip'
        identity = q.inspect_model(model, config, 'C', seed, target, self.revision)
        events = [e for e in self.completed_events if e['event'] == 'checkpoint' and e.get('model') == str(model)]
        if (len(events) != 1 or events[0]['model_sha256'] != identity['model_sha256']
                or events[0]['metadata_sha256'] != identity['metadata_sha256']):
            raise RuntimeError('Completed checkpoint identity changed or is missing')
        meta = q.read_json(model.with_suffix('.json'))
        if meta.get('resume_sha256') != q.sha256(parent) or Path(meta['resume_from']).resolve() != Path(parent).resolve():
            raise RuntimeError('Completed training parent changed')
        arguments = ['-m', 'combat_rl.train', '--run-dir', directory, '--resume', parent, '--steps', target]
        if first and recipe != 'continue':
            arguments += ['--finetune-config', frozen]
        elapsed = self.job(name+'-train', arguments, 0)
        self.rate = max(self.rate*0.8, elapsed/(meta['timesteps']-prior['timesteps']))
        return str(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parents', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--video-deps', required=True, type=Path)
    parser.add_argument('--training-source', required=True, type=Path)
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--predecessor-pid', required=True, type=int)
    parser.add_argument('--handoff-token', required=True)
    args = parser.parse_args()
    args.max_hours = 18  # Historical manifest value; remaining() is deliberately unlimited.
    worker = ContinuationWorker(args)
    try:
        worker.run()
    except Exception as error:
        q.write_json(worker.control/'status.json', dict(phase='failed', error=repr(error), ready=worker.ready))
        raise


if __name__ == '__main__':
    main()
