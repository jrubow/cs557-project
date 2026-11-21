# Policy optimization after the controlled comparison

Written 2026-11-20. The user authorized queuing optimization after comparison.
The supervised follow-on worker implements the automatic stages described below;
the existing comparison queue retains its frozen source and settings.

**Completed 2026-11-21:** the worker finished all 24 supervised jobs in 5.99 hours.
C was selected by the predeclared rule after B/C finished nearly tied across three
training seeds. The 2M continuation passed confirmation: mean validation waves
increased from 2.46 to 2.76 and kills from 5.02 to 6.00, with positive wave changes
on all three seeds and no five-wave successes. The lower-learning-rate candidate
failed screening. The chosen three C/2M checkpoints and original nine 1M models
were locked before the separately authorized held-out test batch. No test-driven
parameter or checkpoint changes are part of this plan.

PPO already updates policy weights during every training run. This plan covers
the additional experimental work needed to improve the learned behavior: decide
whether it needs more experience, better training settings, or a targeted change
supported by observed failures. The aim is the strongest validated policy within
the project deadline and compute budget; global optimality is not a realistic claim.

## Objective and baseline

Keep ordinary evaluation fixed: the same five zombie waves, random starts, arena,
equipment, action timing, carried health, and episode limit. Rank policies by
five-wave completion rate, then mean completed waves, then mean kills. Report
damage dealt/taken, survival time, and health entering each wave as diagnostics.
Raw reward is unsuitable for comparing conditions with different reward functions.

The first training seed averaged 2.02 / 2.72 / 2.16 waves for A / B / C; none of
the 150 validation episodes completed five waves. B is the initial lead, pending
the other two training seeds. The rendered B episode cleared three waves and died
in wave four. That clip suggests useful questions about later-wave survival, but
does not by itself identify the cause of failure.

Finish all nine original runs at the frozen 1M budget before selecting a condition.
Keep their models, CSVs, source revision, and manifest as the controlled comparison.
Choose the condition using the mean of its three training-seed scores. Use seed 0
for initial optimization screening, fixed in advance, rather than choosing the
luckiest checkpoint. Report optimized results separately from the original A/B/C
comparison, including all additional training and tuning cost.

## Ordered experiments and decisions

1. **Diagnose behavior before changing it.** Evaluate the selected condition with
   decision traces on all 50 validation scenarios, and record scenarios
   50,000-50,004 for manual inspection. Inspect health entering each wave, damage
   exchanged per kill, time within attack range, aim error, movement near walls,
   and ineffective attack requests. Compare these with PPO entropy, approximate
   KL, clipping fraction, and value-fit logs. Add diagnostic fields if necessary;
   attack requests alone do not establish whether sword cooldown or targeting
   caused a missed hit. Record a specific failure hypothesis before each change.

2. **Test more training first.** Continue the selected condition's seed-0 model
   from 1M to 1.5M and then 2M total decisions, keeping its saved settings. Evaluate
   50 validation episodes at both milestones and retain the best checkpoint.
   Stop that branch at 2M or sooner if it clearly regresses. If neither milestone
   meets the screening gate below, treat extra training alone as insufficient
   within this budget. Do not automatically extend every A/B/C model.

3. **Screen at most two targeted setting changes.** Start fresh seed-0 candidates
   at the original 1M budget so each has a matching baseline. Default hypotheses
   are a lower learning rate (`3e-4` to `1e-4`) for unstable updates, or a changed
   entropy coefficient (`0.01` to `0.005` for persistent action noise, or `0.02`
   for premature action collapse). Select the direction from diagnostics before
   launching; test one change per candidate. These values are starting hypotheses,
   not known optimal settings. Compare each candidate against the original 1M
   model, not against a 2M continuation. Avoid a large parameter grid.

4. **Use task changes only when evidence warrants them.** A clear failure to learn
   later waves may justify a longer training-only curriculum; repeated inactivity
   or reckless trades may justify a reward ablation. Such a candidate replaces
   one of the two screening slots, rather than expanding the search. Keep ordinary
   evaluation unchanged. Any new observation, action, network architecture, or
   recurrent policy requires a separate implementation and compatibility check;
   defer these larger changes unless simpler experiments fail and time remains.

5. **Confirm the winner across training seeds.** Select one candidate and its
   training budget on seed 0, then reproduce that exact recipe for seeds 1 and 2.
   For more-training candidates, continue each seed's own original model. For
   changed settings, train fresh models. If the chosen recipe exceeds 1M, also
   obtain unchanged-setting controls at that same budget for all three seeds;
   otherwise label the result as a combined settings/budget improvement. Preserve
   every seed's result, including failures, and keep the original policy if the
   improvement does not replicate.

6. **Freeze, test, and show the result.** Lock the chosen recipe, checkpoint
   selection rule, and comparison set before using the held-out test scenarios.
   Evaluate all three seeds of each final recipe on the same 50 test scenarios,
   report each seed and the aggregate with uncertainty, and record representative
   validation episodes for inspection. Test results are for reporting, not another
   round of tuning. Keep the human contextual benchmark and final writing time.

## Promotion rules and time budget

Use deterministic evaluation on validation seeds 50,000-50,049 throughout the
bounded search. A candidate passes the screening gate if it improves five-wave
completion by at least 6 percentage points (3/50 episodes), or improves mean
completed waves by at least 0.25 without reducing completion rate. Break ties
using mean kills, then lower compute cost. These are practical project thresholds,
not statistical significance claims. Repeat a borderline comparison because
Minecraft AI is not bit-exact even with the same scenario seed.

For confirmation, require the same gate on the average of the three seed scores,
with an improvement in the qualifying metric on at least two seeds and no decline
in aggregate completion rate. Report the spread and per-seed differences; 150
episodes from three trained models are not 150 independent training runs. If
confirmation fails or is incomplete, label the candidate exploratory.

Cap this phase at **18 additional simulator-hours**, including diagnostics,
screening, controls, and confirmation; reserve at least six of those hours for
confirmation. Stop screening sooner if the selected confirmation runs will need
more time. The completed 1M runs took about 79-104 minutes each before evaluation;
later training can take longer, so recalculate estimates from measured throughput.
Stop optimization at least **72 hours before the actual submission deadline**, or
when the compute cap is reached, whichever comes first. Preserve the strongest
validated checkpoint; more training is not assumed to improve it monotonically.

## Execution and provenance

- Run one simulator at a time, after the current queue finishes. Different ports
  do not isolate the shared CraftGround game files. Use the existing supervised
  Windows runner, timeouts, status files, and descendant cleanup.
- Write optimization outputs to a new root such as `runs/optimization/`; never
  overwrite the frozen comparison. Give each candidate an immutable config and
  source revision, parent-checkpoint hash where applicable, and a ledger entry
  with hypothesis, training seed, actual steps, runtime, validation metrics,
  promotion decision, and artifact paths. Record abandoned and failed candidates.
- Existing `train --resume --steps N` supports continuation with the same saved
  config; `N` is the target total, not additional steps. It restores the model and
  optimizer, but starts new episodes and is not a bit-exact uninterrupted replay.
  Changed learning rates, entropy settings, or inputs cannot currently be passed
  through resume: use fresh candidate runs. Copy/preserve the baseline checkpoint
  and use a new output directory for each continuation milestone.
- The evaluator currently uses the validation seed range in the saved config.
  A separate confirmation scenario range would require an explicit evaluator
  feature; do not silently change model configs or use test seeds for tuning.
- `scripts/queue_optimization_windows.py` implements the handoff, milestone
  evaluations, selection, and an append-only experiment ledger in a separate root.

## Automatic queue and limits

Launch the worker from a clean, dedicated checkout with `--comparison-root`,
`--output-root`, the predecessor's `--comparison-pid` and `--comparison-created`
timestamp, `--comparison-manifest-sha256`, `--video-deps`, and a timezone-aware
`--cutoff-utc`. It defaults to an 18-hour optimization cap and a 24-hour wait cap.
The waiting time does not consume the optimization budget. The optimization clock
includes all active wall time, which is stricter than counting simulator time alone.
The current desktop launch uses a conservative absolute cutoff of November 24,
2026 at 00:00 Indianapolis time, ahead of the approximate submission deadline.

The worker waits for successful batch completion and predecessor exit, then checks
all nine models, configs, budgets, validation scenario ranges, and training and
evaluation supervisor records. It refuses failures, remaining descendants, changed
source/manifests, or reused output paths. A new optimization directory is required;
failed queues are inspected before any explicit restart. `STOP_AFTER_CURRENT` in
either output root stops the optimizer before its next simulator job.

The automated sequence is baseline traces and five predetermined videos; 1.5M
and 2M continuations; up to two fresh 1M setting candidates; and confirmation of
one selected recipe on seeds 1 and 2. The 2M recipe repeats the same 1.5M restart
boundary during confirmation. Seed-0 screening scores average two independent
50-episode validation passes for both baseline and candidates, rather than only
repeating a favorable borderline result. Seed-level aggregates weight all three
training seeds equally. The confirmation gates above still apply.

Candidate rules use medians of the last 100 logged training updates: learning rate
`1e-4` if approximate KL is at least `0.03` or clipping fraction at least `0.30`;
entropy coefficient `0.005` if joint action entropy divided by its theoretical
maximum is at least `0.60`, or `0.02` if it is at most `0.25`. These predeclared
heuristics may select zero, one, or two candidates. The queue saves all diagnostics
and hypotheses before training. It does not automatically invent reward,
curriculum, action, or observation changes: those require manual review of the
saved traces/videos and a separately recorded experiment decision.

At least six hours are reserved for confirmation, increased if observed runtimes
suggest more is needed. Estimated job costs gate launches and the existing
supervisor enforces the remaining time budget. Each job runs sequentially with
explicit process cleanup. The best validated baseline remains the selection until
the candidate passes confirmation. Partial or failed candidates never replace it.

Outputs include `status.json`, `ledger.jsonl`, `comparison-summary.json`,
`diagnostics.json`, `screening.json`, `confirmation.json` when reached, and
`final-selection.json`, plus models, CSVs, logs, and checked MP4s. A successful
confirmation is followed by three predetermined videos of the selected seed-0
model. Final held-out evaluation is a separate phase: this worker never uses the
test split. The human benchmark, manual video review, and report remain follow-up
work after these automated experiments finish.

The general approach follows SB3's advice to evaluate separately, use multiple
training seeds, and consider additional training and hyperparameter tuning:
[SB3 2.7.1 experiment guidance](https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/rl_tips.html).
Learning-rate and entropy controls are documented in the
[PPO parameter reference](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html#parameters).
The concrete thresholds, candidates, and time limits above are project decisions.
