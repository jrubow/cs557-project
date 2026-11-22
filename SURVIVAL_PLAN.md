# Exploratory five-wave optimization

This is a second study after the first test set was observed. The completed
800-episode study stays frozen in [reports/study-20261121](reports/study-20261121).
The objective is repeatable five-wave completion with the existing structured
policy, iron equipment, five-wave arena, and health carry between waves.

**Budget amendment, November 21:** the user removed the overall wall-clock cutoff
so the planned runs can finish. `continue_survival_windows.py` takes over after
the current supervised job finishes naturally, reuses verified completed jobs,
and executes only the remaining finite protocol. It keeps the original frozen
training source and manifest, with a separate amendment recording this change.
The video worker waits without a deadline. Individual six-hour job timeouts,
failure checks, and process cleanup remain; the 10:25 a.m. overall cutoff is
superseded after the handoff. No in-flight model training is restarted.

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

One simulator runs at a time. The original launch had an 18-hour active budget
and confirmation reserve; those values remain in its immutable historical
manifest. The user's later budget amendment above removes the overall cutoff.
The finite training targets and selection rules are unchanged. At the earlier
measured rate the full 5M path was roughly 16-18 hours; actual completion depends
on throughput and the selected target. `status.json` and `ledger.jsonl` provide
live progress; create `STOP_AFTER_CURRENT` in the output root to stop between jobs.
Failure or requested-stop states do not claim confirmation.

```powershell
python scripts/queue_survival_windows.py --parents <original-optimization-models> `
  --output <new-output-root> --video-deps <video-dependencies> --max-hours 18
```

Checkpoints and full logs stay outside Git. Compact results from the completed
first study are published; this new study's measurements must be reported
separately with their additional training budgets and post-test exploratory status.

## Requested demonstration videos and review gate

On November 21 the user requested a video of a complete five-wave victory after
the current optimization finishes, followed by a continuous-spawn demonstration.
`scripts/queue_demos_windows.py` runs from a separate frozen checkout. It waits
for both the survival worker's terminal status and actual process exit, verifies
all started supervisors cleaned up, and checks the shared Minecraft runtime is
idle. It does not change the running training worker or start a second simulator.

First use the selected best seed-0 development checkpoint and check the three
already-planned fixed recordings. If none is a verified full victory, record new
attempts, preferring that checkpoint's known winning development scenarios. Keep
all attempts and record their outcomes. Stop at the first video that verifies
five completed waves, fifteen kills, terminal success, correct model hash, and
matching encoded frame counts. Export annotated and clean MP4s under easy names.
This is deliberately a selected winning demonstration, not an unbiased win-rate
sample or a replay. Never alter the task, fabricate a victory, or discard failures
to make it look successful. There are at most thirty new attempts with a ninety-
minute active worker cap, reserving time for the endless demo. If none succeeds,
report that honestly and preserve the footage for investigation.

Then record one fixed continuous-spawn demonstration using the same frozen
checkpoint and scenario 51000. Begin with one zombie and add one every 2.0 game
seconds (40 ticks), regardless of how many remain. Use ordinary random arena
spawn positions, the same iron equipment and action controls, no regeneration or
between-wave healing, and no five-wave termination. End at death or a ten-minute
game-time recording limit; a surviving run is censored, not a victory. Report
kills, survival time, final health, damage, kills/minute, peak simultaneous enemies,
and issued spawn commands. The video shows time, HP, kills, and enemies alive.

This separate stress task keeps the original 156-value policy input: its wave
feature is fixed at the last trained stage (5), and only the nearest five visible
enemies fill the existing slots. Additional enemies do not expand the policy's
field of view or observation. New continuous-spawn results are not pooled with
the five-wave benchmark. The final five-wave test scenarios stay untouched.

**Stop for the user's review before calibration or training.** The first footage
is a baseline with no policy updates. Review targeting, attack timing, movement
near walls, response to multiple enemies, and how health is lost. If the user then
wants calibration, compare a few spawn intervals (for example 4, 3, and 2 seconds)
on shared development seeds to measure the arrival rate the policy can sustain.
Freeze the reviewed cadence before tuning a copy of the policy. Preserve this
baseline and measure gains in kills/survival under the same conditions, while
checking for regressions on the original five-wave task. No cadence sweep,
additional policy training, or final test is automatically authorized by this
demonstration worker.
