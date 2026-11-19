# Learning to Survive: Minecraft combat

CS557 project: a small PPO agent learns combat in a controlled CraftGround arena.
This is an implementation of the midterm plan, with the agreed smaller action set.

**Status:** code and six simulator-independent checks are complete. Real Minecraft
startup, combat smoke checks, PPO training, and performance have **not** been run
yet. Run the smoke command below on the compute machine before spending training
time. There are no experimental results in this repository.

## Get the first training run going

Use Linux, Python 3.11, Java 21, and an OpenGL display. The first CraftGround launch
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

`n_envs=1` is intentional: changing ports alone does not isolate CraftGround's
Gradle/runtime/options/world files. To use several machines, use separate clones
and separate Python environments. Parallel workers on one host need separate
runtime directories first; this initial implementation does not configure them.

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

## Work split and remaining gates

- `env.py`: Minecraft commands, resets, waves, termination, history.
- `task.py`: action/observation encoding and reward deltas.
- `train.py`: PPO/checkpoints; `evaluate.py`: fixed scenarios and throughput.
- `policies.py` / `human.py`: baselines; `report.py`: CSV summaries and figures.
- `smoke.py`: short real-simulator checks; `tests/`: six tiny checks requiring only NumPy/Gymnasium.

Suggested remaining ten-day sequence: first get a real smoke/benchmark/pilot on
compute; use the next day to fix integration and calibrate on validation; reserve
the middle five days for the controlled runs; use the final three for held-out
evaluation, human sessions, plots, and writing. Reduce budget before cutting the
evaluation/writing window. Equipment generalization, mixed enemies, RGB policies,
and video export are extensions after the core comparisons work.

Local verification performed: six task/state-machine tests, CLI/syntax checks, and
small synthetic checkpoint/summary checks. Those synthetic values are not saved as
project results. No Minecraft build, real PPO update, GPU test, or training was run
on the initial laptop.

API references: [CraftGround 2.7.4](https://pypi.org/project/craftground/2.7.4/),
[pinned environment source](https://github.com/yhs0602/CraftGround/blob/848a34519247ea68ce878e675381bd8d62084dce/src/craftground/environment/environment.py),
[headless setup](https://yhs0602.github.io/CraftGround/installation/headless.html),
[Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html).
