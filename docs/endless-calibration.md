# Exploratory batch survival and critical-hit evidence

Authorized November 22, 2026, after the user reviewed a timed-spawn demonstration
with four kills and requested cleared batches and longer survival. This is a new
task; it does not replace or get pooled with the original A/B/C five-wave study.
All original checkpoints and results remain intact.

## Execution and task definitions

`scripts/queue_endless_windows.py` runs from a clean frozen checkout. It waits
for the exact continuation controller PID/creation time to exit, requires both
the main and continuation statuses to be `complete`, and validates cleanup of
every supervised predecessor job. Each subsequent job checks for a Minecraft
runtime or occupied port. A failed/stopped predecessor does not start this queue.
There is **no overall wall-clock deadline**. Training jobs have six-hour runaway
protection and frequent saved checkpoints; evaluation/recording jobs have a
90-minute safeguard. `STOP_AFTER_CURRENT` stops between jobs. Outputs are never
silently overwritten or reused after failure.

The parent is the final development-selected `best_seed0` from the preceding
survival experiment. It remains an exploratory parent even if the preceding
three-seed confirmation fails; that confirmation status remains in its own record.

- Timed mode: one initial zombie, then one every 2, 4, or 6 game seconds.
- Batch mode: k initial zombies, with k in {1, 2, 3, 5}. Spawn a new batch only
  after no living zombies remain and the kill counter confirms all k kills.
- Original arena, iron equipment, action pulses, FOV and history remain the same.
  No between-batch healing, reset, or manual aiming. Health carries over.
- Timed mode uses wave feature 5; batch mode uses k, always within the original
  observation range. Policy observations remain 156 dimensions. Sound telemetry
  and RGB never enter the policy input or reward.
- All exploratory episodes end on death or 300 simulated seconds. Episodes at
  the cap are right-censored; their times are lower bounds on possible survival.
  The cap defines an episode, not an overall training deadline.

## Calibration, optimization and confirmation

1. Evaluate all seven profiles on ten scenarios, 52000–52009. Choose the largest
   batch k with median survival at least 60 seconds and median kills at least 10.
   If none qualifies, choose greatest median kills, then median seconds, then k.
   Freeze the task before training; comparisons across k describe difficulty,
   not improvement of the policy.
2. Measure the selected batch task and timed-2s reference on 52100–52119.
3. Train a copy of the parent for cumulative additional targets of 250k, 500k,
   1M and 2M decisions, adjusted for rollout rounding. Warm-start weights and
   optimizer, learning rate 1e-4, other PPO settings and reward weights inherited.
   Each full batch clear receives the existing wave-clear reward. Episode
   summaries from the old task are cleared when loading. This is one training
   seed; it is not a new three-seed confirmatory experiment.
4. Evaluate each checkpoint on the same twenty development scenarios. A candidate
   qualifies only if mean kills increase by at least max(2, 10% of baseline)
   and mean survival does not decrease. Among qualifiers choose mean kills,
   then mean survival. Keep the parent if none qualifies. After at least 500k,
   stop if kills fall below 80% of baseline without better survival.
5. Lock selection, then evaluate parent and selected on fresh scenarios
   52200–52219. Report whether the gain holds; do not select again on these.
   Evaluate both on original five-wave development scenarios 51000–51049 and
   the selected policy on timed-2s as separate transfer checks. Keep the original
   policy available for the five-wave task if adaptation hurts that task.
6. Record parent batch episode 52000, selected batch episodes 52000–52002, and
   selected timed-2s episode 52000. These are fixed-seed demonstrations, not
   selected wins. Preserve all outcomes and per-tick data. HUD includes k,
   kills, remaining health, alive enemies and critical sound cues.

No original test scenarios 100000–100049 or reserved five-wave test scenarios
200000–200049 are used for this development. New exploratory numbers are
reported separately, with capped-episode counts and underlying episode records.

## Critical-hit measurement and learning history

The pinned Minecraft 1.21 language asset maps
`subtitles.entity.player.attack.crit` to `Critical attack`. CraftGround's pinned
`MinecraftSoundListener.kt` retains one sound entry per subtitle key, advances
its age each tick and resets age on a new event. `CombatTelemetry` reads every
raw simulation tick and counts advancing event birth times (tick minus age),
excluding cues present at episode initialization and distant sounds. Lingering
subtitles count once; identical sounds in the same tick may coalesce. This is
critical-attack **sound evidence**, not a direct server-side hit counter.

Before interpretation, `combat_rl.crit_probe` runs a grounded strike and a
falling strike against an immobile zombie. The sensor is marked validated only
if both cause damage, the grounded control has no crit cue and the falling
control has a crit cue. These scripted controls change positioning/action timing
and are labeled calibration, never policy performance. If validation fails,
counts remain raw diagnostics and absent cues do not demonstrate absent crits.

Replay C seed 0 checkpoints nearest 10k, 50k, 100k, 300k and 500k, plus 1M, 2M and
survival10 3M, 4M and 5M on matched scenarios 53000–53009 (original five-wave task).
Record damage events, attack pulses, jumping, airborne/descending attack ticks,
critical cues, cues per 100 attack pulses and per game minute. Later reward/PPO
adaptation is a distinct training stage, not a single-factor causal experiment.

This identifies the **earliest sampled checkpoint showing the behavior**, not
the exact training step when it was learned. Ten episodes per checkpoint can
miss rare behavior. Existing training did not save raw sound cues, so there is
no retrospective count of every training hit. A jumping or attacking action
alone is not counted as a crit. No critical-hit bonus is added during this phase.
The common original initialization favored attacking with steady aim/movement
and no jump; it did not explicitly prescribe jump-attacks or critical-hit timing.

Implementation evidence: `combat_rl/telemetry.py`, `combat_rl/crit_probe.py`,
`combat_rl/survival_task.py`; runtime source
`craftground_runtime_mc121/src/main/java/com/kyhsgeekcode/minecraftenv/MinecraftSoundListener.kt`;
Minecraft 1.21 client asset `assets/minecraft/lang/en_us.json`.
