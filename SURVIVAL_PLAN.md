# Exploratory five-wave optimization

This is a second study after the first test set was observed. The completed
800-episode study stays frozen in [reports/study-20261121](reports/study-20261121).
The objective is repeatable five-wave completion with the existing structured
policy, iron equipment, five-wave arena, and health carry between waves.

## Starting policy and evidence

Start from C at 2M steps, seed 0. It led pre-test validation at 2.99 completed
waves and 6.91 kills per episode (seed 2 was close at 2.96 waves). This choice is
based on validation, not the now-observed test scores. Parent seeds 1 and 2 are
retained for confirming a recipe. Original checkpoint files are never overwritten.

The three existing C 2M diagnostic recordings died after 3, 2 and 2 completed
waves. Health entering wave 3 was 8.96, 6.20 and 7.58 out of 20. The first entered
wave 4 with 4.82 health. This supports investigating survival but does not prove
the cause of damage or that a stronger penalty will help. Excessive damage
penalties may produce avoidance, so wave completion and kills determine selection.

## Frozen protocol

- First verify a roughly 10k-step real-simulator fine-tuning preflight. Its
  checkpoint is excluded from selection; all trials restart from the 2M parent.
- Fresh development scenarios: 51000-51049, 50 deterministic episodes per check.
  Re-evaluate the parent seed 0 twice and each other parent once. Repeat the
  selected seed-0 evaluation before confirmation to reduce rollout noise.
- Reserve 200000-200049 for a new final test after selection is locked. The
  worker never runs that test or reads the old test scores. Fresh scenario seeds
  are within the same fixed arena/world distribution, not new worlds or tasks.
- Compare three recipes from the same 2M seed-0 checkpoint at 2.5M and 3M total
  steps: unchanged continuation; survival10; survival25. Both survival recipes
  use learning rate 0.0001, gamma 0.995, and damage-taken weight 10 or 25 (previously
  0.0003, 0.99 and 1.5). Other settings, all actions, observations and dynamics
  stay fixed. These are joint recipes, not single-factor causal ablations.
- Warm starts restore policy and optimizer state. Changed schedules and rollout
  discount are applied before the model is reconstructed. Fine-tuning requires
  an explicit config and a new directory. Initial action bias is not reapplied.
  Episodes restart at resume boundaries; resuming is not bit-exact uninterrupted
  training. Confirmations repeat the same boundaries.
- Rank by five-wave success rate, then mean completed waves, then mean kills.
  Never select by shaped return. Keep all checkpoints and unsuccessful results.
  Stop a recipe if it loses more than 0.5 mean waves without any success gain.
- Extend the best non-regressing 3M recipe to 4M and, if budget permits, 5M.
  Preserve earlier better checkpoints. A candidate needs at least +0.25 waves
  without losing success rate, or +0.06 success rate, against the parent before
  spending on other-seed confirmation.
- Confirm the selected recipe from the corresponding 2M parents for seeds 1 and
  2, with identical settings and resume milestones. Require the same mean gain
  and a positive change on the qualifying metric in at least two of three seeds.
  If confirmation fails, keep the prior three-seed recipe as the established
  result and label the candidate seed-0 checkpoint exploratory.
- Save three ordinary rendered development episodes (51000-51002). These are
  fixed scenarios, not cherry-picked victories. Five-wave success in training
  is diagnostic; repeated development success and a later locked test are
  needed to substantiate the goal. No five-wave success is promised.

## Runtime and monitoring

`scripts/queue_survival_windows.py` runs from a clean dedicated checkout and
requires a new output root. The manifest freezes source, parent hashes, settings,
scenario lists and budget. Every child is supervised; failed output or leftover
tracked processes stops the queue. The existing native Windows shutdown error
remains documented and is separate from Python exit/result validation.

One simulator runs at a time. The active budget is at most 18 hours. Extensions
reserve time for confirming both other seeds and recording episodes. At the
previous measured rate, the full 5M path is roughly 16-18 hours; an earlier stop
or a 4M selection shortens it. This November 21 run leaves the planned final
72-hour evaluation/report window intact. `status.json` and `ledger.jsonl` provide
live progress; create `STOP_AFTER_CURRENT` in the output root to stop between jobs.
Budget/failure stops are recorded explicitly and do not claim confirmation.

```powershell
python scripts/queue_survival_windows.py --parents <original-optimization-models> `
  --output <new-output-root> --video-deps <video-dependencies> --max-hours 18
```

Checkpoints and full logs stay outside Git. Compact results from the completed
first study are published; this new study's measurements must be reported
separately with their additional training budgets and post-test exploratory status.
