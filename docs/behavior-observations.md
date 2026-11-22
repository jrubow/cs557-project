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
censored intervals; summaries with opportunity counts; checkpoint charts; and
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
