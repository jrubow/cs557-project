# Learning to Survive: Minecraft combat

CS557 project: a small PPO agent learns combat in a controlled CraftGround arena.
This is an implementation of the midterm plan, with the agreed smaller action set.

**Next exploratory phase:** [batch survival and critical-hit analysis](docs/endless-calibration.md)
waits for the current survival confirmations and recordings to finish. It compares
cleared batches of 1/2/3/5 zombies with timed arrivals, then fine-tunes a copy for
up to 2M additional decisions. Passive critical-sound telemetry is validated with
scripted controls and measured across saved C checkpoints. This phase has no
overall time cutoff and uses development scenarios only; it does not replace the
original five-wave study or its test results.

The follow-up [behavior observation protocol](docs/behavior-observations.md)
compares aiming, hit-and-retreat, attack timing, focus fire, enemy reacquisition
and wall escapes across the same saved checkpoints. Focus-fire diagnostics include
first-damage-to-death time per zombie, switches before killing, simultaneous
wounded enemies and incomplete timers. It adds passive traces and annotated
videos after batch optimization, without changing policy inputs or rewards.

**Status:** all nine controlled A/B/C runs are complete on native Windows with
Python 3.11 and Java 21. Each fresh seed-0/1/2 model trained for 1,000,448 decisions
under identical frozen code/config/budgets, then ran 50 shared validation scenarios
with ordinary starting positions. Nineteen offline checks and the earlier real
Minecraft combat smoke/integration checks passed.

| Condition | Mean waves completed | Mean kills | Five-wave successes |
| --- | ---: | ---: | ---: |
| A | 1.34 | 2.33 | 0/150 |
| B | 2.45 | 4.96 | 0/150 |
| C | 2.46 | 5.01 | 0/150 |

These are validation results averaged across three training seeds. A/seed 2
performed poorly (0.02 mean waves); all seeds remain in the reported comparison.
B and C are essentially tied. The predeclared ranking selected C for optimization.
Extending C through 1.5M to 2M improved mean validation waves on all three seeds;
the confirmed aggregate was 2.76 waves and 6.00 kills, with no five-wave success.
The baseline for that confirmation averaged 2.46 waves and 5.02 kills, including a
second seed-0 validation pass. A lower-learning-rate candidate did not improve.
This additional-budget result is separate from the controlled 1M A/B/C comparison.

The optimization queue finished all 24 supervised jobs on November 21, 2026,
including three new recordings of the selected policy. Models, recordings, and
results remain outside Git. The selected recipe and all checkpoint hashes are
locked before held-out evaluation; test results are not used for further tuning.

The [policy optimization plan](OPTIMIZATION_PLAN.md) defines the completed phase:
diagnose failures, test additional training, screen a small number of targeted
changes, and confirm improvements across training seeds before final testing.
It includes selection rules and a compute/deadline limit. A separate
`scripts/queue_optimization_windows.py` worker can wait for the comparison to
finish, run bounded validation-only optimization, and save its selected models.
The original comparison retains its frozen source, settings, and 1M-step budgets.

`scripts/run_test_evaluation_windows.py` evaluates the nine original models and
three selected C/2M models on test seeds 100,000-100,049. Four reference evaluations
cover random/scripted controllers with full and restricted observations (800
episodes total). Full-view references use B's reward settings; their actions do
not consume rewards, and A/B have identical task dynamics and observations.
Reported combat metrics, rather than raw reward, support comparisons. The runner
preserves training-seed groups, separates 1M and 2M budgets, verifies scenario and
model identities, and performs no training or post-test model selection.

Held-out PPO results (mean +/- SD across three trained models, each evaluated on
50 test scenarios):

| Recipe | Mean waves completed | Mean kills | Five-wave successes |
| --- | ---: | ---: | ---: |
| A, 1M | 1.31 +/- 1.11 | 2.27 +/- 1.95 | 0/150 |
| B, 1M | 2.45 +/- 0.24 | 4.97 +/- 0.77 | 0/150 |
| C, 1M | 2.49 +/- 0.22 | 5.09 +/- 0.57 | 0/150 |
| C, 2M extension | 2.79 +/- 0.22 | 6.08 +/- 0.77 | 0/150 |

The C extension improved test waves by 0.48, 0.30, and 0.12 for training seeds
0, 1, and 2, respectively (mean paired change +0.30 waves). This supports improved
combat progress within the existing task, but no tested PPO model completed all
five waves. These are descriptive results from three trained models per recipe;
150 episodes are not 150 independent training runs. The test split assesses new
spawn/yaw scenarios in the same arena with the same equipment and enemies, not
new worlds or equipment generalization. All checkpoints remain unchanged.

The four references each ran 50 test episodes. Random full/restricted controllers
completed no waves and made no kills. Scripted full averaged 2.54 waves and 5.38
kills; scripted restricted averaged 2.22 waves and 4.62 kills. Neither completed
five waves. The full test batch finished on November 21, 2026 in 42.8 minutes:
all 16 supervised evaluations exited zero, all 800 rows passed scenario/identity
checks, and all model hashes remained unchanged. No controller completed all five
waves in this batch. The human contextual benchmark remains pending.

## Get the first training run going

Use Python 3.11, Java 21, and an OpenGL display (Linux or native Windows). The first CraftGround launch
downloads and builds Minecraft/native dependencies; allow time and network access.
Start with one simulator and at least several GB of free RAM (the JVM defaults to
a 3 GB heap). Minecraft throughput, not this small MLP, is likely to determine
the training budget. An A100 allocation alone does not establish a working OpenGL
display.

```bash
git clone https://github.com/jrubow/cs557-project.git
cd cs557-project
conda env create -f environment.yml
conda activate cs557
python -m pip install -e '.[train]'
java -version                 # Must select Java 21, not the system Java 25.
python -m unittest discover -s tests -v
```

On an Ubuntu server, an administrator may need to install the display/build
libraries first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git xvfb xauth mesa-utils \
  libgl1-mesa-dev libegl1-mesa-dev libglew-dev libglu1-mesa-dev \
  xorg-dev libglfw3-dev libpng-dev zlib1g-dev libflite1
```

Activate the environment before launching. If JNI/Java discovery fails, set
`JAVA_HOME` to the actual Java 21 installation and clear the failed native build
cache with `python -c 'import craftground; craftground.clear_native_build_cache()'`.
Do not change Java versions halfway through a build.

For the first headless run, the supplied wrapper uses Xvfb and software OpenGL:

```bash
bash scripts/headless.sh python -m combat_rl.smoke
bash scripts/headless.sh python -m combat_rl.evaluate \
  --benchmark-steps 500 --output runs/benchmark.json
bash scripts/headless.sh python -m combat_rl.train \
  --condition B --seed 0 --steps 10000 --run-dir runs/pilot
bash scripts/headless.sh python -m combat_rl.evaluate \
  --model runs/pilot/final.zip --split validation --episodes 10 \
  --output runs/pilot/validation.csv
```

The smoke check tests real reset, movement/turning, received damage, sword damage,
kills, and a wave transition. It fails rather than reporting a partial pass. It
uses controlled enemy positioning to isolate these mechanisms; it does not prove
that a learned policy will succeed. Its default wall timeout is 15 minutes; a
cold first build may need `--timeout-seconds 1800`.

Software rendering is a fallback. Once this works, benchmark a configured hardware
display/VirtualGL following CraftGround's setup guide. For VirtualGL, set
`environment.use_vglrun` to `true` in a config, and launch under that configured
display instead of the software-rendering wrapper. The default policy runs on CPU;
`--device cuda` is available, but measure before assuming it is faster.

If startup stalls at Minecraft's accessibility screen, the pinned runtime may
need `onboardAccessibility:false` in its game `options.txt`. Inspect that screen/log
before changing the installed runtime. Keep the pinned package versions together.

### Native Windows

Use an isolated Conda environment and the normal desktop graphics driver. Install
Visual Studio Build Tools with the C++ desktop workload/Windows SDK if a compiler
is not already available. This path was exercised with Visual Studio 18 Build
Tools, Java 21, and the pinned Python packages; it does not need WSL or Xvfb.
Choose a reasonably short environment prefix, with Windows long paths enabled.
The Linux `environment.yml` and `headless.sh` are not Windows setup scripts.

```powershell
conda env create -f environment-windows.yml
conda activate cs557-windows
python -m pip install -e ".[train]"
python -m unittest discover -s tests -v
python scripts/windows_run.py --timeout 1800 --log runs/smoke.log -- -m combat_rl.smoke --timeout-seconds 1800
python scripts/windows_run.py --timeout 1200 --log runs/benchmark.log -- -m combat_rl.evaluate --benchmark-steps 500 --output runs/benchmark.json
python scripts/windows_run.py --timeout 1800 --log runs/pilot.log -- -m combat_rl.train --condition B --seed 0 --steps 10000 --run-dir runs/pilot
python scripts/windows_run.py --timeout 1800 --log runs/validation.log -- -m combat_rl.evaluate --model runs/pilot/final.zip --split validation --episodes 10 --output runs/pilot/validation.csv
```

The wrapper selects Java and native-library paths from the Python interpreter's
Conda environment, keeps Gradle's cache there, records combined logs and a
`.status.json`, and monitors only its own process descendants. It applies an
external wall timeout and cleans up tracked descendants, including on normal
exit. The smoke command's Unix SIGALRM alone does not bound Windows execution.
Exit code 124 means the external timeout fired. Use new log/output paths on reruns;
run only one simulator at a time. Set a longer timeout for longer training runs;
forced timeouts cannot guarantee a new checkpoint.

The pinned runtime currently reports native exit `-1073740791` from Gradle during
Windows shutdown, after successful checks/results. Waiting for graceful exit did
not resolve it. This remains an upstream shutdown limitation, not a clean JVM
exit or evidence that every integration path is tested. Preserve the logs and
check the Python result/status and descendant cleanup separately. Accessibility
onboarding did not block the tested installation; no runtime source edits were
needed to build or complete the combat checks.

## Experiment

| Condition | Reward | Information |
| --- | --- | --- |
| A | Kills, completed waves, death, elapsed time | All arena zombies |
| B | A plus damage dealt/received | All arena zombies |
| C | Same as B | Zombies within a 100-degree horizontal cone |

The arena has a bedrock floor and enclosing walls, normal difficulty, and fixed
night/clear weather. Natural spawning, natural regeneration, mob griefing, and
mob loot are disabled. The player gets an unbreakable iron sword/armor and fixed
food via saturation. Zombies are adults, unarmed, unarmored, and cannot pick up
items. Spawn positions vary; health carries between waves. Death, completion, or
3,000 decisions ends an episode. Evaluation uses waves of 1, 2, 3, 4, then 5 zombies.

The training curriculum uses the same **total environment-step** schedule for all
conditions: one wave before 100k steps, three waves from 100k, five from 300k. A
stage change takes effect on the next episode. One decision normally advances four
Minecraft ticks, about 0.2 seconds of game time. Wall time is separate. The default
budget is 1M decisions per run; PPO rounds to complete 1,024-step rollouts.

An optional `environment.encounter_curriculum` can make **training-only first
encounters** easier. Each stage is `[start_step, min_spawn_radius,
max_spawn_radius, initial_yaw_error_degrees]`. The player begins aimed roughly at
the zombie's torso when yaw error is below 180 degrees, then controls every camera
change normally. There is no continuous target lock. Later waves use the original
spawn rules. An empty list (the default) preserves the original environment.
Evaluation always uses the original 4–10 block spawn radius and random initial
orientation, regardless of this curriculum.

The experimental `calibration-encounter*.json` configs progressively relax starts
at 0/50k/100k steps and restore the original geometry/orientation at 200k. A live
Minecraft check verified close/aligned early starts, full initial health and zero
setup rewards, plus identical unassisted late-training/evaluation starts. The
same curriculum must be used for A, B, and C if selected. The reward-5 candidate
only increases damage-dealt shaping from 1 to 5; A's outcome rewards are unchanged.
These files are calibration candidates, not a declaration that learning succeeds.

`training.initial_action_bias` optionally initializes a fresh PPO policy's action
logits to favor standing still, neutral yaw/pitch, no jump/sprint, and attack.
The default is zero; `calibration-steady.json` uses 2.5 to reduce unproductive
initial camera drift. Every action remains available, sampled during training,
and learned normally. This is a common initialization prior, not scripted control
or a change to observations/rewards. It is applied only to new models, never when
resuming, and must be identical across final A/B/C runs.

`configs/default.json` contains the common settings. JSON overrides can specify
only changed keys, and unknown keys fail early. Choose coefficients/budgets on
validation, then hold them fixed for the final A/B/C comparison. A short single-wave
pilot is for discovering setup/learning problems, not evidence about five-wave play.

Actions are `MultiDiscrete([3,3,5,3,2,2,2])`:

| Dimension | Choices |
| --- | --- |
| Move | still, forward, backward |
| Strafe | still, left, right |
| Yaw | -30, -10, 0, +10, +30 degrees |
| Pitch | -10, 0, +10 degrees |
| Jump / sprint / attack | off/on, independently |

Camera changes apply once per decision. Attack is pressed on the first tick and
released on subsequent ticks, because CraftGround uses mouse-button transitions.
Minecraft still applies its weapon cooldown. The policy aims; there is no target
lock. Drop, use, sneak, inventory, and crafting are excluded.

Each frame has seven player features (health, x/z, sine/cosine yaw, pitch, wave)
and five enemy slots (relative right/forward/up, health, presence). All are scaled
into [-1,1]. Four frames and their preceding actions form a 156-value observation.
Enemy slots are sorted by distance after visibility filtering. Condition C uses
a horizontal cone, **not** pixel vision or occlusion; prior visible observations
remain in history. Hidden enemy positions, health, counts, and reward counters are
not added to the policy input.

Rewards are calculated per game tick. Default outcome reward is `1 * kills +
3 * waves - 5 * death - 0.001 * ticks`. Shaping adds `damage_dealt / 20 -
1.5 * damage_taken / 20`. Damage comes from the player's Minecraft statistics,
converted from tenths to health points; it is not inferred from enemy disappearance.
Reset statistics establish a fresh baseline. Death takes precedence over a
simultaneous final kill. A wave needs both no live zombies and the expected
player-attributed kills.

## Training, resume, and evaluation

After the pilot and benchmark, choose a feasible common budget. The benchmark
reports decisions/sec and hours per million decisions; include PPO updates,
evaluation, and at least 25% scheduling margin. Nine 1M-step runs require roughly
`9 * hours_per_million_steps`, before that overhead. Shorten all conditions equally
if the measured budget does not fit.

```bash
# Run only after choosing the common config/budget; sequential by design.
STEPS=1000000 bash scripts/headless.sh bash scripts/run_experiments.sh

# Resume the latest checkpoint after an interruption; --steps is the target TOTAL.
bash scripts/headless.sh python -m combat_rl.train --run-dir runs/B_seed0 \
  --resume runs/B_seed0/interrupted.zip --steps 1000000

# A model's saved config and condition are used automatically.
bash scripts/headless.sh python -m combat_rl.evaluate \
  --model runs/B_seed0/final.zip --split test --output results/B_seed0_test.csv
bash scripts/headless.sh python -m combat_rl.evaluate \
  --policy random --condition B --split test --output results/random_B_test.csv
bash scripts/headless.sh python -m combat_rl.evaluate \
  --policy heuristic --condition C --split test --output results/heuristic_C_test.csv

python -m combat_rl.report results/*_test.csv --output-dir results/summary \
  --training-dirs runs/A_seed0 runs/B_seed0 runs/C_seed0
```

Repeat final evaluation for every trained seed/condition. Baselines use the same
scenario seeds and compact actions. Run the random and heuristic controllers in
each reported condition. A/B have the same observations but different returns;
compare waves/success/damage, not raw reward across conditions.

Each model gets a `.zip` and matching `.json` containing config, condition,
training seed, timestep count, and runtime/source metadata. Keep both files.
Checkpoints are saved about every 50k steps, at completion, on Ctrl-C/SIGTERM, and
when a training exception can be handled. Resume restores model, optimizer, and
step counter; it starts fresh episodes and discards any partial rollout, so it is
not a bit-exact continuation. `SIGKILL` cannot save a new checkpoint.

Training logs include episode metrics and PPO progress CSV. Evaluation writes one
CSV row per episode and a metadata JSON. Validation seeds start at 50,000; test
seeds start at 100,000 (50 episodes each by default). These reproduce Python spawn
layouts and yaw, not Minecraft's internal AI randomness after fast reset. Use the
same seed sets for paired scenarios and report variability across independent
training seeds. The summary averages episodes within runs before calculating
across-run means/SD; one run has an undefined SD, not zero variability.

For behavior inspection, add `--trace-output runs/my_run/validation-trace.csv` to
an episode evaluation command. This optional CSV records each action, pre-action
player state, nearest-enemy distance/bearing, approximate torso pitch error, and
actual reward/statistic deltas. Enemy geometry is diagnostic ground truth only;
it is never added to the policy input (including condition C). Distance and angle
thresholds are aiming proxies, not a Minecraft hit test. Trace recording is
excluded from the throughput benchmark. The sum of trace deltas should match
the corresponding episode CSV metrics.

### Watch a rendered episode

Install the video extra with `python -m pip install -e ".[train,video]"`. Once
all active simulator runs have finished, record a fresh episode from a saved model:

```powershell
python scripts/windows_run.py --timeout 1800 --log runs/record-B.log -- -m combat_rl.record_episode --model runs/B_seed1/final.zip --output-dir videos/B_seed1-validation50000
```

This reads Minecraft's offscreen RGB framebuffer; the native desktop window is
not the video source. It saves a clean `episode.mp4`, an `episode-annotated.mp4`
with health/wave/action labels, event PNGs, per-frame data, and episode metadata.
Playback is 20 frames per second, matching Minecraft game time; the labeled video
adds a two-second terminal hold. Requested attack actions still use the existing
press/release pulses. Capture starts after reset setup and ends at termination.

The recorder uses the first validation scenario by default and deterministic PPO
actions. `--scenario-seed` can select another validation seed; test seeds are
rejected. Display-only overrides increase capture resolution from 160x90 to
640x360 and show the native sword/crosshair/HUD for inspection; use
`--hide-game-hud` to retain the training HUD setting. RGB and diagnostic labels
never enter the structured policy observations. This is a fresh
diagnostic rollout, not a replay of earlier CSV results: Minecraft AI randomness
can differ despite identical Python scenario seeds. Frame counts and final
damage/kill counters are checked against the recorded episode.

A real B/seed-1 diagnostic on validation scenario 50,000 recorded 54.0 game
seconds, six kills, and three completed waves before death during wave four.
The 1,081 raw frames and 1,121 annotated frames (including the end hold) decoded
successfully at 20 fps; timing and final counters matched the episode record.
Decoded frames were visually inspected and contained changing Minecraft scenes.
This fresh rollout differs from the earlier same-seed CSV, as expected from the
runtime's non-bit-exact AI behavior.

`n_envs=1` is intentional: changing ports alone does not isolate CraftGround's
Gradle/runtime/options/world files. To use several machines, use separate clones
and separate Python environments. Parallel workers on one host need separate
runtime directories first; this initial implementation does not configure them.

On Windows, `scripts/run_experiments_windows.py` accepts `--config`, `--run-root`,
`--steps`, `--seeds`, and `--conditions` to run selected jobs sequentially. Use it
only after choosing the common configuration on validation. It freezes the full
config and a source/budget manifest in the run root, writes `status.json`, enforces a per-training-process
timeout, and evaluates each completed checkpoint on validation. It stops on a
failed process or a changed source revision. Keep that checkout unchanged while
the queue is running. Later queue invocations in that root must match the pinned
source revision, configuration, training budget, and validation episode count.
Creating `RUN_ROOT/STOP_AFTER_CURRENT` prevents the next
run from starting. Existing nonempty run directories are rejected; recover a
failed run explicitly with the documented checkpoint resume command, then queue
the remaining jobs. It never runs the test split automatically.

`configs/comparison.json` records every selected setting explicitly, with a 1M
step budget and checkpoints approximately every 10k decisions. To start one fresh
comparison run from a clean, frozen checkout:

```powershell
python scripts/run_experiments_windows.py --config configs/comparison.json --run-root runs/comparison --conditions B --seeds 1 --steps 1000000 --episodes 50
```

This selects one run. Preserve its configuration and source revision when adding
other seeds/conditions later; calibrations that resume an earlier model remain
separate from these fresh, equal-budget runs.

## Human contextual benchmark

On a machine with a desktop display:

```bash
python -m pip install -e '.[train,human]'
python -m combat_rl.human --condition B --episodes 10 --split test \
  --participant p01 --output results/human_p01_B_test.csv
```

This is a heading-up diagram of the **same structured observation**, not the usual
Minecraft interface. W/S move, A/D strafe, arrows aim (Ctrl for fast yaw), Space
jumps, Shift sprints, and F attacks. Escape omits the incomplete episode and saves
completed ones. Humans use the same scenario prefix and action space. Report this
interface and the smaller sample explicitly; it is contextual, not a matched
sample-efficiency comparison. Human responses have at least the simulated decision
interval, and a slow simulator may give extra thinking time.

## Further five-wave optimization

The completed 800-episode test data, chart, summaries and model/source provenance
are published in [reports/study-20261121](reports/study-20261121). No evaluated
controller cleared all five waves. The next exploratory study starts from C 2M
seed 0, chosen by its pre-test validation result, and compares continuation with
survival-focused fine-tuning. [SURVIVAL_PLAN.md](SURVIVAL_PLAN.md) specifies fresh
development/final-test ranges, bounded training, checkpoint retention and
confirmation across the other seeds. The completed test study remains frozen.

`train --resume <checkpoint> --finetune-config <config> --run-dir <new-directory>`
explicitly permits reward and selected PPO-setting changes while preserving the
task and policy spaces. Ordinary resume retains its original strict behavior.
`evaluate --first-seed <seed>` supports fresh ranges within a split and rejects
validation ranges that cross the configured test boundary.

After that worker exits, `scripts/queue_demos_windows.py` can reuse or capture a
verified five-wave winning video and record a separate continuous-spawn stress
demo. `record_episode --mode endless --spawn-seconds 2 --max-seconds 600` records
one zombie every two game seconds until death or the recording limit. This is
inference only, with the original gear and observation/action spaces; the wave
feature is fixed at 5. The [demonstration plan](SURVIVAL_PLAN.md#requested-demonstration-videos-and-review-gate)
keeps these results separate and stops for user review before calibration/tuning.

The user subsequently removed the overall training deadline. A graceful
`continue_survival_windows.py` handoff validates and reuses finished jobs from
the original output root, keeps its frozen training source, and completes the
remaining finite protocol without an overall time limit. It records a separate
manifest amendment and retains per-job failure protection. The demonstration
worker accepts `--wait-hours 0` to wait for that continuation without a deadline.

## Work split and remaining gates

- `env.py`: Minecraft commands, resets, waves, termination, history.
- `task.py`: action/observation encoding and reward deltas.
- `train.py`: PPO/checkpoints; `evaluate.py`: fixed scenarios and throughput.
- `policies.py` / `human.py`: baselines; `report.py`: CSV summaries and figures.
- `smoke.py`: short real-simulator checks; `tests/`: task/environment boundaries,
  queue handoff/selection, checkpoint adaptation, scenario separation and video
  integrity checks. Tests use the Windows training environment, including one
  SB3 save/reload check; none of these unit tests launch Minecraft.

Suggested remaining ten-day sequence: first get a real smoke/benchmark/pilot on
compute; use the next day to fix integration and calibrate on validation; reserve
the middle five days for the controlled runs; use the final three for held-out
evaluation, human sessions, plots, and writing. Reduce budget before cutting the
evaluation/writing window. Equipment generalization, mixed enemies, and RGB policies
are extensions after the core comparisons work. Diagnostic video recording is
available for inspecting the current structured-observation policies.

Initial laptop verification covered six task/state-machine tests, CLI/syntax,
and synthetic checkpoint/summary checks. On the Windows desktop, the real
Minecraft build and combat smoke check passed, followed by the 500-step benchmark,
10,240-step CPU PPO pilot (70 training episodes), checkpoint reload, and ten
validation episodes using seeds 50,000–50,009. The pilot's full supervised process
took 142.1 seconds including startup/cleanup. Its 133-second PPO timing includes
startup; later rollouts took roughly 5–6 seconds per 1,024 steps. Neither training
nor validation completed a wave. At that stage, neither held-out test evaluation
nor the nine-run sweep had been launched. No CUDA policy benchmark has been performed.

The follow-up B calibration resumed the saved pilot to 100,352 total steps (980
PPO update epochs), adding 90,112 decisions in 498.6 seconds including startup and
cleanup. Its 196 additional completed training episodes had no kills. On ten
validation scenarios it survived to every 600-game-second time limit, with zero
kills and mean damage dealt 0.41, compared with 2.02 for random and 118.66 for the
scripted controller. The scripted controller completed 2–3 waves per episode,
though none completed all five. These are small validation samples from one PPO
training seed, not final test results.

The 30,000-decision PPO trace showed attack selected 86.8% of the time, nearest
enemy distance below three blocks on 0.97%, and absolute pitch above 45 degrees on
73.75%. Trace lengths and reward/damage/kill sums matched the episode CSVs for
both PPO and the scripted controller. This supports evasion and ineffective
aiming as the current learning failure; it does not establish a best remedy.

Four fresh B/seed-0 calibrations then tested stronger damage-dealt shaping,
easier early encounters, both together, and both plus steady-action initialization.
The first three still produced zero validation kills at 100,352 steps. The last
candidate (`calibration-steady.json`) made 340 kills in 791 completed training
episodes and cleared at least one wave in 8/10 ordinary validation episodes.
It averaged 0.9 waves, 1.1 kills, and 33.1 damage dealt; all episodes ended in death.
Its trace spent 54.82% of decisions within three blocks of an enemy and only
0.24% with absolute pitch above 45 degrees. Trace totals matched every episode.
These exploratory runs support continuing the combined configuration; they do
not isolate each component's causal contribution or establish seed robustness.

The selected candidate then resumed to 300,032 total steps (2,930 PPO update
epochs), adding 199,680 decisions in 1,006.9 seconds including startup/cleanup.
Its 616 additional completed training episodes made 1,231 kills and completed
the three-wave training target nine times. Starting-position assistance was
fully removed at 200k; the ordinary wave curriculum remains common across A/B/C.
On 50 validation scenarios (50,000–50,049), every episode cleared at least one
wave, averaging 1.9 waves, 3.14 kills, and 76.52 damage dealt. All ended in death,
with no five-wave success. The 300k checkpoint has only just reached the five-wave
training stage. Its trace counts and reward/damage/kill sums matched all 50
episode records. The continuation measured 198.3 decisions/second including
startup/cleanup; this is measured multi-wave throughput from one training seed.

Preserve these calibration results separately from fresh controlled runs. The
100k combined calibration took 531.6 seconds including startup/cleanup; planning
roughly two hours for each 1M-step run leaves margin for slower later waves.
Reserve separate time for evaluation, baselines, human sessions, analysis, and
writing. Use identical encounter/wave curricula, initialization, and budget for
all final A/B/C conditions, and keep the held-out test scenarios untouched until
the configuration is fixed. All nine controlled runs subsequently completed with
those frozen settings; current validation and test status is summarized above.

API references: [CraftGround 2.7.4](https://pypi.org/project/craftground/2.7.4/),
[pinned environment source](https://github.com/yhs0602/CraftGround/blob/848a34519247ea68ce878e675381bd8d62084dce/src/craftground/environment/environment.py),
[headless setup](https://yhs0602.github.io/CraftGround/installation/headless.html),
[Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html).
