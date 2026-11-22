"""Second, exploratory study: warm-start C and optimize five-wave completion.

Use a clean dedicated checkout. One supervised simulator at a time. The old test
study is closed; this worker never reads test scores or runs a new test split.
"""
import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from combat_rl.train import finetune_changes
from scripts.queue_optimization_windows import (
    average, clean_supervisor, confirmed, evaluation_score, inspect_model,
    qualifies, rank, read_json, sha256, write_json,
)

DEV_FIRST = 51000
FINAL_TEST_FIRST = 200000
EPISODES = 50
RECIPES = ("continue", "survival10", "survival25")


class BudgetStop(RuntimeError):
    pass


def recipe_config(parent_config, recipe):
    config = copy.deepcopy(parent_config)
    if recipe not in RECIPES:
        raise ValueError("Unknown recipe")
    if recipe != "continue":
        config["training"].update(learning_rate=0.0001, gamma=0.995)
        config["reward"]["damage_taken"] = 10.0 if recipe == "survival10" else 25.0
    finetune_changes(parent_config, config)
    return config


def severe_regression(score, baseline):
    return score["success_rate"] <= baseline["success_rate"] and score["waves"] < baseline["waves"] - 0.5


class Worker:
    def __init__(self, args):
        self.args = args
        self.source = Path(__file__).resolve().parents[1]
        self.root = args.output.resolve()
        self.started = time.monotonic()
        self.rate = 0.005  # Measured prior continuation: about 200 decisions/s.
        self.jobs = 0
        self.state = {}
        self.revision = self.git("rev-parse", "HEAD")
        self.parent_models = [args.parents.resolve() / f"more_2000000_seed{s}/final.zip" for s in range(3)]
        self.parents = [read_json(p.with_suffix('.json')) for p in self.parent_models]
        self.original_hashes = {str(p): sha256(p) for p in self.parent_models}
        self.best = None

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.source, text=True).strip()

    def source_check(self):
        if self.git("rev-parse", "HEAD") != self.revision or self.git("status", "--porcelain"):
            raise RuntimeError("Source changed; refusing to mix implementations")
        if any(sha256(Path(p)) != digest for p, digest in self.original_hashes.items()):
            raise RuntimeError("Original checkpoint changed")

    def remaining(self):
        return self.args.max_hours * 3600 - (time.monotonic() - self.started)

    def status(self, phase, **extra):
        self.state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=time.monotonic() - self.started,
                          remaining_seconds=max(0, self.remaining()), jobs_completed=self.jobs, **extra)
        write_json(self.root / "status.json", self.state)

    def ledger(self, event, **extra):
        import json
        with (self.root / "ledger.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=event, **extra)) + "\n")

    def estimate(self, decisions):
        return max(300, decisions * self.rate * 1.15 + 90)

    def reserve(self, target):
        # Reserve both other seeds, four evaluation jobs, and videos/cleanup.
        return self.estimate(2 * (target - 2_000_000)) + 1800

    def job(self, label, arguments, estimate, reserve=0):
        if (self.root / "STOP_AFTER_CURRENT").exists():
            raise BudgetStop("STOP_AFTER_CURRENT requested; current job finished")
        self.source_check()
        available = self.remaining() - reserve - 30
        if available < estimate:
            raise BudgetStop(f"Not enough time for {label}; preserved confirmation reserve")
        # Never launch into another Minecraft instance using the shared runtime.
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", 8000)) == 0:
                raise RuntimeError("Port 8000 is occupied; only one simulator is allowed")
        log = self.root / 'logs' / (label + '.log')
        if log.exists():
            raise RuntimeError("Refusing to reuse job output")
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.args.video_deps.resolve())
        command = [sys.executable, str(self.source / 'scripts/windows_run.py'),
                   '--timeout', str(min(21600, available)), '--log', str(log), '--', *map(str, arguments)]
        self.status('running', current_job=label)
        self.ledger('job_started', label=label, command=command)
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=self.source, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        while process.poll() is None:
            self.status('running')
            time.sleep(5)
        record = clean_supervisor(log.with_suffix('.status.json'))
        if process.returncode:
            raise RuntimeError(f"Supervisor failed: {label}")
        self.jobs += 1
        self.ledger('job_completed', label=label, supervisor=record)
        return time.monotonic() - started

    def evaluate(self, name, model):
        output = self.root / 'evaluations' / (name + '.csv')
        self.job(name, ['-m', 'combat_rl.evaluate', '--model', model, '--output', output,
                       '--split', 'validation', '--episodes', EPISODES, '--first-seed', DEV_FIRST], 300)
        meta = read_json(Path(model).with_suffix('.json'))
        score = evaluation_score(output, model, meta['config'], 'C', meta['seed'], first_seed=DEV_FIRST)
        self.ledger('evaluation', name=name, model=str(model), score=score, csv_sha256=sha256(output))
        return score

    def train(self, recipe, seed, target, parent, first=False, reserve=0):
        name = f'{recipe}_{target}_seed{seed}'
        directory = self.root / 'models' / name
        if directory.exists():
            raise RuntimeError('Training directory already exists')
        prior = read_json(Path(parent).with_suffix('.json'))
        config = recipe_config(prior['config'], recipe) if first else copy.deepcopy(prior['config'])
        config['training']['total_timesteps'] = target
        frozen = self.root / 'configs' / (name + '.json')
        write_json(frozen, config)
        arguments = ['-m', 'combat_rl.train', '--run-dir', directory, '--resume', parent, '--steps', target]
        if first and recipe != 'continue':
            arguments += ['--finetune-config', frozen]
        parent_hash = sha256(parent)
        elapsed = self.job(name + '-train', arguments, self.estimate(target-prior['timesteps']), reserve)
        model = directory / 'final.zip'
        identity = inspect_model(model, config, 'C', seed, target, self.revision)
        new = read_json(model.with_suffix('.json'))
        if (sha256(parent) != parent_hash or new.get('resume_sha256') != parent_hash
                or Path(new['resume_from']).resolve() != Path(parent).resolve()):
            raise RuntimeError('Resume provenance mismatch')
        actual = new['timesteps'] - prior['timesteps']
        self.rate = max(self.rate * 0.8, elapsed / actual)
        self.ledger('checkpoint', recipe=recipe, target=target, seed=seed,
                    parent=str(parent), parent_sha256=parent_hash, **identity)
        return str(model)

    def save_selection(self, status, **extra):
        self.source_check()
        write_json(self.root / 'selection.json', dict(status=status, best_seed0=self.best,
            development_scenarios=[DEV_FIRST, DEV_FIRST+49], reserved_test_scenarios=[FINAL_TEST_FIRST, FINAL_TEST_FIRST+49],
            new_test_used=False, old_study_closed=True, **extra))

    def initialize(self):
        self.source_check()
        if self.root.exists():
            raise RuntimeError('Use a completely new output root')
        for seed, (path, meta) in enumerate(zip(self.parent_models, self.parents)):
            inspect_model(path, meta['config'], 'C', seed, 2_000_000, meta['runtime']['git_revision'])
            if meta['config'] != self.parents[0]['config']:
                raise RuntimeError('Parent settings differ')
        self.root.mkdir(parents=True)
        write_json(self.root / 'manifest.json', dict(source_revision=self.revision,
            source=str(self.source), frozen_utc=datetime.now(timezone.utc).isoformat(),
            parents=[dict(model=str(p), model_sha256=sha256(p), metadata_sha256=sha256(p.with_suffix('.json'))) for p in self.parent_models],
            parent_config=self.parents[0]['config'], recipes={r: recipe_config(self.parents[0]['config'], r) for r in RECIPES},
            development_scenarios=list(range(DEV_FIRST, DEV_FIRST+EPISODES)),
            reserved_test_scenarios=list(range(FINAL_TEST_FIRST, FINAL_TEST_FIRST+EPISODES)),
            max_hours=self.args.max_hours, selection='success rate, then completed waves, then kills',
            note='Exploratory second study after the original test was observed. Same arena/gear/actions; no healing or scripted aim. Survival recipes jointly change damage penalty, learning rate and discount, so they are not single-factor ablations. See SURVIVAL_PLAN.md.'))

    def run(self):
        self.initialize()
        try:
            # Verify real simulator warm-start/config integration before the long
            # queue. This small checkpoint is excluded from selection and trials.
            self.train('survival10', 0, 2_010_000, str(self.parent_models[0]), first=True)
            baselines = [self.evaluate(f'parent_seed{s}', str(p)) for s, p in enumerate(self.parent_models)]
            baselines[0] = average([baselines[0], self.evaluate('parent_seed0_repeat', str(self.parent_models[0]))])
            baseline = baselines[0]
            self.best = dict(recipe='parent', target=2_000_000, model=str(self.parent_models[0]), score=baseline, milestones=[])
            candidates = []
            for recipe in RECIPES:
                parent = str(self.parent_models[0])
                for target in (2_500_000, 3_000_000):
                    parent = self.train(recipe, 0, target, parent, first=target == 2_500_000, reserve=self.reserve(3_000_000))
                    score = self.evaluate(f'{recipe}_{target}_seed0', parent)
                    candidate = dict(recipe=recipe, target=target, model=parent, score=score,
                                     milestones=[t for t in (2_500_000, 3_000_000) if t <= target])
                    if rank(score) > rank(self.best['score']):
                        self.best = candidate
                    self.save_selection('screening')
                    if severe_regression(score, baseline):
                        self.ledger('recipe_stopped', recipe=recipe, reason='More than 0.5 waves below baseline without a success gain')
                        break
                else:
                    candidates.append(candidate)
            # Continue the best non-regressing 3M recipe. Keep earlier bests too.
            if candidates:
                current = max(candidates, key=lambda c: rank(c['score']))
                if rank(current['score']) >= rank(baseline):
                    for target in (4_000_000, 5_000_000):
                        required = self.estimate(target-current['target']) + self.reserve(target) + 600
                        if self.remaining() < required:
                            self.ledger('extension_skipped', target=target, reason='Confirmation reserve')
                            break
                        model = self.train(current['recipe'], 0, target, current['model'], reserve=self.reserve(target))
                        score = self.evaluate(f"{current['recipe']}_{target}_seed0", model)
                        current = dict(recipe=current['recipe'], target=target, model=model, score=score,
                                       milestones=current['milestones']+[target])
                        if rank(score) > rank(self.best['score']):
                            self.best = current
                        self.save_selection('extending')
                        if severe_regression(score, self.best['score']):
                            break
            # Re-evaluation protects against retaining one lucky noisy rollout.
            if self.best['recipe'] != 'parent':
                repeat = self.evaluate('selected_seed0_repeat', self.best['model'])
                self.best['score'] = average([self.best['score'], repeat])
            scores = [self.best['score']]
            models = [self.best['model']]
            if self.best['recipe'] != 'parent' and qualifies(self.best['score'], baseline):
                for seed in (1, 2):
                    model = str(self.parent_models[seed])
                    for index, target in enumerate(self.best['milestones']):
                        model = self.train(self.best['recipe'], seed, target, model, first=index == 0)
                    models.append(model)
                    scores.append(self.evaluate(f'confirmed_seed{seed}', model))
                status = 'confirmed_improvement' if confirmed(scores, baselines) else 'confirmation_failed'
            else:
                status = 'no_confirmed_gain'
            retained = models if status == 'confirmed_improvement' else [str(p) for p in self.parent_models]
            self.save_selection(status, candidate_models=models, candidate_scores=scores, baseline_scores=baselines,
                                retained_models=retained, retention_note='Exploratory best_seed0 remains available even if recipe confirmation fails.')
            try:
                self.record(self.best['model'])
                video_status = 'complete'
            except BudgetStop as error:
                # A video budget stop must not erase completed confirmation.
                video_status = str(error)
                self.ledger('recording_stopped', reason=video_status)
            self.status('complete', current_job=None, selection_status=status, video_status=video_status)
        except BudgetStop as error:
            self.save_selection('budget_stopped', reason=str(error), retained_models=[str(p) for p in self.parent_models])
            self.status('budget_stopped', reason=str(error), current_job=None)
        except Exception as error:
            self.status('failed', error=repr(error))
            raise

    def record(self, model):
        # Save three ordinary development episodes, not a cherry-picked success.
        for seed in range(DEV_FIRST, DEV_FIRST+3):
            output = self.root / 'videos' / f'selected-{seed}'
            self.job(f'video-{seed}', ['-m', 'combat_rl.record_episode', '--model', model,
                     '--output-dir', output, '--scenario-seed', seed], 300)
            meta = read_json(output / 'episode.json')
            if meta['model_sha256'] != sha256(model) or (output / 'INCOMPLETE.txt').exists():
                raise RuntimeError('Recording incomplete or model mismatch')
            sys.path.insert(0, str(self.args.video_deps.resolve()))
            import imageio_ffmpeg
            for name, key in [('episode.mp4', 'raw_frames'), ('episode-annotated.mp4', 'annotated_frames')]:
                count, duration = imageio_ffmpeg.count_frames_and_secs(str(output / name))
                if count != meta[key] or abs(duration-count/20) > 0.06:
                    raise RuntimeError('Recording frame count mismatch')
            self.ledger('video_verified', path=str(output), metrics=meta['metrics'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parents', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--video-deps', required=True, type=Path)
    parser.add_argument('--max-hours', type=float, default=18)
    args = parser.parse_args()
    if not 1 <= args.max_hours <= 18:
        parser.error('Budget must be between 1 and 18 hours')
    Worker(args).run()


if __name__ == '__main__':
    main()
