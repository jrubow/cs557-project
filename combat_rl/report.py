"""Summarize held-out evaluations, preserving training seeds as independent runs."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def number(value):
    if str(value).lower() in ("true", "false"):
        return float(str(value).lower() == "true")
    return float(value)


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--training-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--no-plots", action="store_true", help="Write CSV summaries without matplotlib")
    args = parser.parse_args()
    groups = defaultdict(list)
    expected_seeds = {}
    expected_targets = {}
    runs = []
    metrics = ["waves_completed", "success", "kills", "damage_dealt", "damage_taken",
               "elapsed_seconds", "successful_completion_seconds"]
    for path in args.evaluations:
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            parser.error(f"Empty evaluation: {path}")
        identity = {(row["policy"], row["condition"], row["split"], row["training_seed"], row["model"])
                    for row in rows}
        if len(identity) != 1:
            parser.error(f"Mixed run identities in {path}")
        policy, condition, split, training_seed, model = identity.pop()
        targets = {int(row["wave_target"]) for row in rows}
        if len(targets) != 1:
            parser.error(f"Mixed evaluation wave targets in {path}")
        target = targets.pop()
        if split in expected_targets and target != expected_targets[split]:
            parser.error(f"Wave targets differ within split '{split}': {path}")
        expected_targets[split] = target
        seeds = {int(row["scenario_seed"]) for row in rows}
        if len(seeds) != len(rows):
            parser.error(f"Repeated scenario seeds in {path}")
        # Human sessions are a smaller contextual benchmark, not the automated test suite.
        seed_group = (split, policy == "human")
        if seed_group in expected_seeds and seeds != expected_seeds[seed_group]:
            parser.error(f"Scenario seeds differ within split '{split}': {path}")
        expected_seeds[seed_group] = seeds
        group = groups[(policy, condition, split)]
        if any(row["training_seed"] == training_seed for row in group):
            parser.error(f"Duplicate training seed in {policy}/{condition}/{split}: {training_seed}")
        record = {"policy": policy, "condition": condition, "split": split,
                  "training_seed": training_seed, "model": model, "episodes": len(rows),
                  "wave_target": target}
        for metric in metrics[:-1]:
            record[metric] = mean(number(row[metric]) for row in rows)
        record[metrics[-1]] = mean(number(row["elapsed_seconds"]) for row in rows if number(row["success"]))
        group.append(record)
        runs.append(record)

    summaries = []
    for (policy, condition, split), records in sorted(groups.items()):
        summary = {"policy": policy, "condition": condition, "split": split,
                   "independent_runs": len(records), "episodes_per_run": records[0]["episodes"],
                   "wave_target": records[0]["wave_target"]}
        for metric in metrics:
            values = [row[metric] for row in records if row[metric] is not None]
            summary[metric + "_mean"] = mean(values)
            summary[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_run.csv", runs)
    write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "notes.txt").write_text(
        "Each evaluation file is one independently trained model (or one random baseline).\n"
        "Means first average episodes within each run, then average the run means.\n"
        "SD is across training runs; blank SD means fewer than two runs, not zero variability.\n"
        "Success is the fraction completing wave_target waves (default five). Completion time excludes failures.\n"
        "Scenario seeds are identical within each split. Validation and test remain separate.\n"
        "Human sessions use their smaller shared seed set; runs then mean participants, not trained models.\n"
        "Training curves describe curriculum encounters and are not final five-wave evaluations.\n")
    if args.no_plots:
        print(f"Saved summaries to {args.output_dir}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{row['policy']} {row['condition']}\n{row['split']}" for row in summaries]
    figure, axes = plt.subplots(1, 2, figsize=(max(8, len(labels) * 1.7), 4))
    for axis, metric, title, limit in zip(axes, ["waves_completed", "success"],
                                        ["Waves completed", "Target-wave success rate"],
                                        [max(row["wave_target"] for row in summaries), 1]):
        values = [row[metric + "_mean"] for row in summaries]
        errors = [row[metric + "_sd"] or 0.0 for row in summaries]
        axis.bar(labels, values, yerr=errors, capsize=4)
        axis.set_ylabel(title)
        axis.set_ylim(0, max(limit, max(v + e for v, e in zip(values, errors))) * 1.05)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Held-out evaluation (error bars: SD across independent runs)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "evaluation.png", dpi=180)
    plt.close(figure)

    if args.training_dirs:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for directory in args.training_dirs:
            metadata = json.loads((directory / "config.json").read_text())
            points = {}
            for path in sorted(directory.glob("logs-*/progress.csv")):
                with path.open() as stream:
                    for row in csv.DictReader(stream):
                        if row.get("time/total_timesteps") and row.get("combat/waves_completed"):
                            points[int(float(row["time/total_timesteps"]))] = float(row["combat/waves_completed"])
            if points:
                steps = sorted(points)
                axis.plot(steps, [points[step] for step in steps],
                          label=f"{metadata['condition']} seed {metadata['seed']}")
                plotted = True
        if plotted:
            axis.set(xlabel="Training environment steps", ylabel="Recent mean waves completed",
                     title="Training encounters (curriculum changes difficulty)")
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(args.output_dir / "learning_curves.png", dpi=180)
        plt.close(figure)
    print(f"Saved summaries and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
