# Policy selection before the reserved test

Locked from development evidence on November 22, 2026, before opening scenarios
200000–200049. The test is reporting-only: its results cannot change these choices.

The final five-wave recipe is **survival10 at 4M decisions**, retaining training
seeds 0, 1 and 2. Its development mean completed waves increased from 2.78 for
the corresponding C/2M parents to 4.01, with improvement in all three seed lineages.
The seed-0 4M checkpoint is the demonstration policy: its two development passes
produced 22/50 and 17/50 wins (39% combined), compared with 17/50 for its 5M
continuation. Seed 1 and seed 2 of the selected 4M recipe achieved 9/50 and 1/50.
The variation is retained in reporting; a successful video is not a win-rate claim.

The **batch +1M seed-0 model** is retained as a separate exploratory demonstration
policy for replenished groups of three zombies. Its development gain was 17.35
to 21.6 kills, but the fresh confirmation was 17.7 to 19.1 kills, below the
predeclared minimum gain of two kills. Survival increased from 89.42 to 93.45
seconds in that confirmation. It is therefore not a confirmed replacement for
the original survival policy. Its five-wave development transfer was 14/50 wins
against the parent's 17/50. The additional behavior sample is diagnostic and
does not override that predeclared confirmation decision.

The reserved test consists of seven models with 50 episodes each: the three
C/2M parents, the three survival10/4M descendants, and the batch +1M seed-0 model.
Every model uses the same original five-wave task and scenarios 200000–200049.
PPO actions are deterministic; Minecraft AI is not bit-exact. Model and metadata
SHA-256 hashes, source revision, evidence hashes and selection rationale are
frozen in the test manifest before the first episode. No policy updates occur.

Report the parent and survival recipes with equal weight per training seed,
paired changes within each seed lineage, and variation across the three models.
The comparison includes both more training and a joint reward/PPO-settings change;
it is not a single-factor causal ablation. Report the single batch model separately.
The original 800-episode A/B/C study and its already-observed test set remain closed.
