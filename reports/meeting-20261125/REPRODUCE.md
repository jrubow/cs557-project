# Use the meeting artifacts

You can watch the included MP4s and read the PDF without installing Minecraft or Python. The following is for collaborators who want to inspect or run the saved models.

## Find the right files

Clone or download branch `codex/windows-validation` of `jrubow/cs557-project`. Commands below start in the repository root, with the project Python environment active. Follow the root [README](../../README.md#native-windows) for native Windows setup and the pinned dependencies. The original runs used Python 3.11, Java 21, CraftGround 2.7.4, runtime-mc121 0.1.0 and Stable Baselines3 2.7.1.

Keep each `final.zip` beside its `final.json`. The ZIP contains the policy; the JSON supplies its settings. Original absolute paths in metadata are provenance, not paths you need to recreate. The [checkpoint index](checkpoint-index.json) gives portable paths relative to this meeting directory, hashes, training seeds and actual decision counts.

The main choices are:

- Five-wave demonstration: `reports/meeting-20261125/checkpoints/survival10_4M_seed0/final.zip`.
- Batch demonstration: `reports/meeting-20261125/checkpoints/batch_plus1M_seed0/final.zip`.
- The other survival seeds are retained for reporting; do not replace them with only the best seed.

## Check the download

This verifies all hashes in the meeting packet without launching Minecraft:

```powershell
python scripts/verify_meeting_packet.py --packet reports/meeting-20261125
```

## Record a new development demonstration

Run only one simulator at a time. The Windows supervisor bounds the process and cleans up its children. Choose new output directories if these already exist.

```powershell
python scripts/windows_run.py --timeout 1800 --log runs/meeting-five-wave.log -- -m combat_rl.record_episode --model reports/meeting-20261125/checkpoints/survival10_4M_seed0/final.zip --scenario-seed 54000 --output-dir runs/meeting-five-wave
```

For batches of three, with passive critical-hit and behavior annotations:

```powershell
python scripts/windows_run.py --timeout 1800 --log runs/meeting-batch.log -- -m combat_rl.record_episode --model reports/meeting-20261125/checkpoints/batch_plus1M_seed0/final.zip --mode batch --batch-size 3 --max-seconds 300 --track-crits --track-behavior --scenario-seed 54000 --output-dir runs/meeting-batch
```

Recording needs the optional video dependencies used by the project, including Pillow, imageio and imageio-ffmpeg. Check the [recording instructions](../../README.md#watch-a-rendered-episode). A fresh rollout is not an exact replay: Minecraft AI is not bit-exact even with the same scenario seed. The included videos are the actual saved demonstrations.

## Inspect or recompute existing results

The original test CSV is [here](../study-20261121/episodes.csv), and each reserved-test model has its own CSV in [the final report directory](../final-study-20261122/). Exact byte-preserving source copies are also in `evidence/cs557-test/evaluations` and `evidence/cs557-reserved-test-20261122/evaluations`.

Each directory under `evidence/behavior` contains `episodes.json`, `metadata.json` and `ticks.csv.gz`. The compressed CSV has the passive per-tick observations. `combat_rl.behavior.aggregate` recomputes the behavior summary from the episode records. Definitions and measurement limits are in [the protocol](../../docs/behavior-observations.md).

Development evidence directories retain the historical selection documents. For example, an old selection file may still say its test was unused because that was true when selection was frozen. The later test manifest and complete status record that it was subsequently run. Do not edit those historical records to look current.

Both test sets, 100000-100049 and 200000-200049, are now observed. Use new development scenarios for demonstrations. Do not rerun or tune against the reserved tests and describe that as a new unbiased test.

## Files that do the work

| File | Job |
| --- | --- |
| `combat_rl/task.py` | Encode observations/actions and calculate reward |
| `combat_rl/env.py` | Connect the task to actual Minecraft |
| `combat_rl/train.py` | Train, save and resume PPO policies |
| `combat_rl/evaluate.py` | Measure frozen policies |
| `combat_rl/record_episode.py` | Save rendered demonstrations |
| `combat_rl/behavior.py`, `focus.py`, `telemetry.py` | Passive behavior measurements |
| `scripts/windows_run.py` | Supervise a native Windows simulator job |
| `scripts/package_meeting_assets.py` | Build portable copies and video encodes from original local outputs |
| `scripts/render_project_walkthrough.py` | Render the editable Markdown guide as PDF |

The packaging script uses the original local output tree. Collaborators only need the exported evidence to inspect results, and the portable model/metadata pairs to run the selected policies.
