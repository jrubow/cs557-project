# Observed combat behavior across learning checkpoints

This inference-only extension follows the critical-hit study and the existing
batch optimization. The current frozen training queue continues unchanged.
No aiming, kiting or movement heuristic is added to policy actions or rewards.
Ground-truth entity state is logged passively for analysis and is never added
to the policy observation.

## Measurements

**Aiming.** Track horizontal bearing and approximate vertical torso error to the
nearest of the same five enemies allowed by the policy's FOV/distance filter.
Count camera turns that reduce error by more than one degree, and report their
opportunity denominator. Compute camera-only improvement with player/target
positions held at their pre-tick values, so a zombie moving into the crosshair
does not count as the policy learning to turn. Also preserve target IDs and
same-target error after the tick; target switches cannot create fake alignment.
Pitch correction is measured separately. A geometric attack band means within
4 horizontal blocks, 15 degrees of yaw and 20 degrees of torso pitch error.
The torso/eye offsets (1.0/1.62) are proxies, not exact hitbox intersections.
The pinned runtime's supplied raycast is not a trustworthy entity crosshair
counter here, so it is deliberately not used as one.

**Kiting / hit-and-retreat.** Use nonoverlapping one-second windows. A retreat
combat window needs at least ten ticks with a visible enemy within four blocks
and 45 degrees, at least ten backward-command ticks, at least six ticks with
more than 0.01 blocks of measured player motion away from that enemy, and at
least one actual attack pulse. Enemy knockback alone cannot satisfy the motion
test: project only player displacement onto the pre-tick player-to-enemy vector.
Within those windows count positive damage dealt with zero damage taken as
an *unanswered retreat window*. Report damage dealt/taken, enemy count and
starting health for retreat, approach, strafe and other windows. These groups
are mutually exclusive in that priority order. Approach/strafe use matching
command and radial/tangential displacement requirements.

This is evidence consistent with kiting, not proof that the player was outside
every enemy's reach. Damage statistics can lag attacks; damage just outside a
window is not counted inside it. Backward input plus knockback from an earlier
hit can also contribute to movement. Retain full tick traces to inspect these
cases, and do not interpret movement-group comparisons as causal experiments.
Four blocks is an analysis band, not a claimed Minecraft attack-reach constant.
Final partial windows are retained and explicitly excluded from full-window
rates and example selection.

**Attack timing and efficiency.** Record actual edge-triggered attack pulses,
within-episode intervals, damage per pulse, airborne attacks, critical sound
cues, and attacks with no currently visible enemy. A request is not a successful
hit. Lack of a currently visible enemy does not prove a wasted action, because
the policy also has observation history. These diagnostics can show changes in
attack discipline without assuming optimal cooldown timing.

**Finding enemies again.** When live enemies exist but none are currently in the
policy's visible slots, time the interval until one becomes visible and measure
camera travel during the search. Record episode-ending intervals as censored
and distinguish enemies disappearing from successful reacquisition. Enemy
movement can contribute to visibility; reacquisition alone is not proof of an
intentional search strategy.

**Focus fire and time to kill.** Track every arena zombie by UUID, including those
outside the policy's field of view. The primary duration is `(first observed
health-zero tick - first observed health-decrease tick) / 20`, in game seconds.
Start at first damage, not spawn or the first attack request. The mean and median
pool completed zombies and include their sample count; retain every zombie record
and per-episode evidence. A lethal first hit has duration zero at the 0.05-second
observation resolution. A lingering corpse counts only once. The runtime's
`ToMessage.kt` supplies the UUID and living-entity health; collect the raw entity
list because the policy's `read_state` deliberately removes health-zero corpses.

Require a healthy 20-HP observation before first damage and uninterrupted entity
observations for the primary kill-time statistic. Zombies initially seen injured
have an unknown earlier damage time and are excluded, with counts retained. A
zombie disappearing is an unknown outcome, even if the aggregate player kill
counter increases: that counter alone cannot identify the dead zombie. Record
unfinished timers when the player dies or the episode cap is reached, and record
lost observations separately. Report initially injured cases, observation gaps,
excluded deaths and health-confirmed deaths alongside the player's kill total.
Completed-only means can favor policies that abandon hard targets, so never
interpret them without the incomplete counts and episode survival/kill results.

For complementary evidence, count successive single-target health-loss events
that stay on the same UUID or switch while the previously damaged zombie remains
alive. Count opportunities only when at least two zombies were alive before the
next event; the previous target must still be observed alive when switching.
Choosing a new enemy after a confirmed death is not a switch-before-kill. A tick
with multiple damaged zombies is ambiguous (for example, a sword sweep), is
reported separately, and breaks the target sequence. Also measure mean wounded
zombies alive and the fraction of multi-enemy ticks with multiple wounded zombies.
These are health-loss proxies, not attributed attacks: environmental damage and
sweeps can contribute. Lower kill time, fewer switches and fewer simultaneous
wounded enemies together can support focus-fire behavior, but do not prove intent
or a causal performance benefit. No new focus-fire reward is introduced.

**Walls and maneuvering.** Measure fraction of time within two coordinate blocks
of an arena wall, incoming damage per second near/away from walls, and how long
it takes to move back to at least three blocks away. This hysteresis prevents
small boundary jitter from inflating escapes. Preserve censored intervals when
the episode ends before an escape, and count escapes within five seconds.
Sidestepping uses actual tangential movement, not just a strafe button press.

## Comparisons and evidence

`scripts/queue_behavior_windows.py` waits for the exact current batch controller
to exit successfully, verifies every started supervisor completed cleanly, and
checks the shared runtime is idle before every job. It runs from a new frozen
checkout with no overall deadline. The existing job timeouts and stop marker
remain available. A failed predecessor stops this analysis rather than starting
a competing Minecraft instance.

- Replay the same ten C seed-0 checkpoints (about 10k through 5M decisions) used
  by the crit study, ten five-wave development scenarios each: 53000–53009.
- Run a seeded uniform-random action reference and a steady-attack reference on
  those scenarios. They are explicit reference controllers, not reconstructed
  initial PPO checkpoints. Original PPO initialization already favored steady
  aim/movement and attack, so an early preference alone is not learned evidence.
- Compare the original five-wave parent and the locked, batch-selected policy
  on twenty fresh batch scenarios 53100–53119. Also evaluate the selected model
  on the original five-wave task. Keep tasks separate in every summary.
- Save three fixed-scenario annotated recordings: original policy on five waves
  (53000), original policy on the chosen batch task (53100), and selected policy
  on that batch task (53100). Show aim-turn correction and unanswered retreat
  counters. These recordings are fresh rollouts, not replays of earlier traces.

Outputs include gzip-compressed per-tick CSVs with entity identities/positions,
pre/post player geometry, actions and damage; complete per-episode windows and
censored intervals; raw post-tick UUID/health maps (including corpses), damaged/dead
UUIDs, complete zombie timers and target-damage events; summaries with opportunity
counts; checkpoint charts; and
the first three qualifying retreat windows per episode for manual examination.
All outcomes remain available, not just illustrative windows. Passive observers
must reproduce environment tick and damage totals exactly.

Compare the trend with early checkpoints and references before describing a
behavior as acquired. Ten episodes can miss rare behavior; better policies
also reach different waves and encounter different enemies. Conditional rates,
episode-level evidence and exposure counts are retained for that reason.
Do not claim intention, exact acquisition time, causation, or statistical
significance from the plot alone. The later survival recipe changes several
training settings together. Original and reserved test splits remain untouched.

## Resuming an interrupted analysis

`scripts/continue_behavior_windows.py` verifies completed supervisor receipts,
metadata/model/trace hashes and recomputed behavior summaries before reusing old
evaluations. It runs only the missing jobs in a fresh directory, retains the failed
attempt, and records each reused output's original revision. Cached successes are
not rerolled. The original completed batch and its cleanup records are rechecked.

The first analysis reached nine complete jobs, then a new backend construction
stalled before any Minecraft child or episode trace appeared. Its 90-minute job
guard exited 124 with no surviving descendants. A precise blocking stack was not
available. Native Windows supervised jobs now skip CraftGround's Unix `/tmp`
orphan-socket routine, which otherwise calls `open_files()` across unrelated
processes before launching Minecraft. Native Windows uses TCP and our supervisor
retains ownership/cleanup of its own process tree. This scoped operational change
does not change the task, actions, observations or reward. It is disabled without
the supervisor's Windows environment flag. Backend construction and episode reset
also log their stages and dump diagnostic stacks if blocked for five minutes.
