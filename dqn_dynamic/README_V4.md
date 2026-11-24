# DQN v4: dynamic delivery experiments

This is an isolated extension of the DQN contribution. It does not change
`env.py`, `shared.py`, the application, Q-Learning, or the teammate's DDQN code.
The original v1/v2/v3 models and results remain historical baselines. The new
models use **standard DQN**, not Double DQN or a dueling network.

## Run from the repository root

Use the project's Python environment with NumPy, pandas and PyTorch installed.
No new third-party dependency is required for the v4 pipeline.

```powershell
python -B -m unittest dqn.test_v4 -v
python -B -m dqn_dynamic.run_experiments --smoke
python -B -m dqn_dynamic.run_experiments --steps 60000 --seeds 11 22 33 --eval-episodes 200
```

`python -B dqn_dynamic/part_b_refine.py` now invokes the same v4 runner. The old functions
in that file are retained only as historical helpers; they are not used by the
new pipeline. Do not use those helpers for new performance claims: their old
evaluation conventions differ from v4.

For a smaller **validation-only pilot**, use:

```powershell
python -B -m dqn_dynamic.run_experiments --steps 5000 --seeds 11 22 33 --eval-episodes 30 --validation-only
```

The full defaults train four candidate configurations on three seeds, then
four additional ablation variants on the same three seeds: 24 runs, each with
60,000 environment interactions. Training is CPU-based, with one Torch thread
by default (`--threads` changes this). A smoke run is an integration check, not
an adequately trained model or evidence of superiority.

## Part A: controlled comparisons

All variants use the same chosen hyperparameters and interaction budget:

| Variant | Features | Training reward | Training conditions |
| --- | --- | --- | --- |
| baseline | v3-style 15D | Legacy movement cost | Dynamic |
| state_only | Deadline/cost 30D | Legacy movement cost | Dynamic |
| reward_only | v3-style 15D | Travel minutes | Dynamic |
| full_dynamic | Deadline/cost 30D | Travel minutes | Dynamic |
| full_static | Deadline/cost 30D | Travel minutes | Static |

This is a controlled comparison inside the **new environment**, not a claim
that the baseline exactly reproduces the historical v3 training loop. It uses
correct finite-task terminal handling and the new training split. The selected
configuration is optimized for `full_dynamic`; the ablations isolate its
components rather than giving each variant its own best possible tuning.

The 30D state contains the 15 original-style inputs plus:

- Elapsed minutes, historical target minutes and remaining target slack.
- Four valid-neighbor flags.
- Four current neighboring travel times.
- Four local travel-time changes across the most recent event update.

Time features use a fixed maximum-trip scale; neighboring times use the maximum
move duration, 8.8 minutes. Coordinates/directions retain v3 normalization.
The state contains no future schedule or hidden full-map input. As in other
local-observation routing tasks, distant congestion and future changes remain
partially observed. This extension does not claim a fully observable MDP.
Boundary flags are inputs, not a hard action mask; all variants can incur wall
penalties, and no pathfinding heuristic is injected into the DQN policy.

## Dynamic simulator assumptions

- Single delivery, 10x10 grid, four movement actions, 60-step task limit.
- Orders supply distance, starting traffic/weather, and historical duration.
- One synthetic congestion map is sampled from the partition's map bank.
- Dynamic conditions update every 8 simulated minutes; stress conditions every
  4 minutes with higher initial hotspot density and more map changes.
- Traffic varies at each event; weather evolves every third event.
- A seeded corridor incident appears at the first event and clears at the
  third. The corridor is derived from the initial endpoints, not the agent's
  chosen path. Other cells can become congested or clear too.
- The complete event schedule is generated at reset from a separate RNG. Every
  policy on a paired episode faces the same conditions at the same clock time.
- A move's travel time is fixed using conditions at departure. Events crossed
  during that move are exposed on arrival. There is no mid-edge interruption,
  waiting action, real-road geometry, or real-time external traffic feed.
- Arriving at the destination terminates successfully. Using the last allowed
  step without delivery terminates in failure and does not bootstrap. A global
  training-budget cutoff does not manufacture a terminal transition.

Historical `Time_taken` is a **benchmark proxy**, not a recorded customer
promise. Weather/traffic schedules are synthetic and must be described that way
in the assignment. Short deliveries may finish before the first change; the
episode table records `events_crossed` to support a change-exposed subgroup.

## Reward and common evaluation objective

Time-based reward per step:

```text
- travel_minutes - 2 * wall_bump
+ 50 on delivery - 0.5 * max(0, arrival_time - historical_target) on delivery
- 20 on task failure
```

The legacy-reward ablation replaces only `travel_minutes` with the original
traffic/weather/hotspot movement-cost formula. For **every** variant, the
evaluator separately computes the same time-based `common_return`. Never
compare the raw training rewards of the two objectives as if they were equal.

Metrics include delivery success, on-time fraction over all episodes, common
return, average elapsed time over all attempts, completed-delivery mean/p95
duration and lateness, hotspot entries, backtracks and wall bumps. Completed-
delivery metrics are NaN if there are no completed deliveries; they must always
be read beside success rate. Failures are never treated as successful zero-
lateness or fast deliveries.

Two non-learning baselines are evaluated on exactly the same episodes:

- **Greedy:** row-first direct routing, ignoring congestion.
- **Current-map time planner:** Dijkstra replanned each step using current
  travel minutes. It sees the full current map, unlike the DQN's local inputs.
  It has no future event knowledge and is not a dynamic optimality oracle.

The equal-weight ensemble combines independently trained full-DQN seeds. It
is reported both alone and with a fixed 1.5 normalized-Q backtracking penalty.
Weights and penalty are declared before test evaluation; no best test seed is
selected. Ensemble inference uses multiple networks, so it has higher compute
cost and should not be credited as a free single-model improvement.

## Part B: selection and generalization

1. Keep the team's deterministic 20% test rows; split the other 80% into 64%
   training and 16% validation. Store original row indices in the manifest.
2. Fit distance scaling on training rows only and reuse it in all environments.
3. Use disjoint map banks: training 0..63, validation 1000..1015, test 2000..2031.
   Validation episode seeds start at 100000, test seeds at 200000.
4. Compare standard-DQN configurations. Two search modes are available:
   - `--search manual` (default): change one parameter at a time — baseline,
     lower learning rate, lower discount, faster target updates.
   - `--search random`: random search with successive halving (see below).
5. Select using average validation success rate across seeds, breaking ties
   with average validation common return. Record the selection before testing.
6. Train ablations with that fixed configuration; report every training seed.
7. Evaluate static, dynamic and stress scenarios. Stress has a faster change
   frequency and higher map density than training.
8. Produce paired differences versus greedy and between full DQN and its
   learned ablation controls, with a crossed bootstrap over
   training seeds and matched episode IDs. With three seeds, uncertainty
   estimates are exploratory; they are not proof of generalization everywhere.

The historical test rows were already inspected in earlier work. They are held
out from the **new** tuning, not a genuinely untouched dataset. New map/event
seeds improve coverage but cannot undo historical test exposure. Also, the
split is by order row, not by rider, city or date. A future external dataset or
predeclared group/time split would provide stronger real-world evidence.

### Random search with successive halving (`--search random`)

The brief says random search is often more efficient than checking every
combination, and that slow-to-train models benefit from smarter search. Both
are implemented in `hparam_search.py`:

```powershell
python -B -m dqn_dynamic.run_experiments --search random --trials 12 --finalists 3 --validation-only
```

- **Search space (8 settings):** learning rate (log-uniform 1e-4..3e-3),
  discount gamma {0.9, 0.95, 0.97, 0.99}, batch size {32, 64, 128}, hidden
  units {64, 128, 256}, target update {250, 500, 1000}, train every
  {1, 2, 4} steps, final epsilon {0.01, 0.05, 0.1}, epsilon decay fraction
  (uniform 0.3..0.8). A grid over this space (4 points per continuous
  setting) would need **15,552** full runs; random search uses 12 trials.
- **Random search:** each trial draws every setting independently, so 12
  trials test 12 different learning rates instead of a grid's 4.
  `--search-seed` makes the draws reproducible.
- **Successive halving:** with the defaults (`--rungs 2 --eta 2`), all 12
  trials train for 15,000 steps on the first training seed and are scored on
  validation; the best half continue to 30,000 steps; the best 3 become
  finalists. That costs 360,000 steps (about six full runs) instead of
  720,000 for training all 12 to the full budget.
- **Same final selection:** the finalists and `baseline_config` then go through
  the normal step 5 selection (every training seed, full budget, validation
  only). Keeping the default settings in that last round shows whether the
  search actually helped.
- **Caveats:** short-budget scores are noisy and can favour settings that learn
  quickly but plateau lower. The epsilon schedule scales with the budget, so a
  15,000-step trial explores for fewer steps. Search uses one seed; only the
  final round uses all seeds.

`random_search.csv` records every trial, round, budget, score and whether it
was promoted; `manifest.json` stores the space, seeds and finalists.

Use `--validation-only` while developing. Do not repeatedly select changes by
looking at test results. A final-test run executes only after the validation
selection has been written, but human discipline is still needed across runs.

## Files and artifacts

- `dynamic_env.py`: isolated dynamics and feature extraction.
- `dqn_v4.py`: standard DQN, training and checkpoint loading.
- `experiments.py`: partitions, scales, map banks and manifests.
- `evaluate_dqn.py`: policies, episode metrics and paired uncertainty.
- `hparam_search.py`: random search space and successive halving.
- `run_experiments.py`: tuning, state/reward ablations, ensembles, evaluation.
- `test_v4.py`: behavioral regression tests.

Each invocation creates a new timestamped directory in `dqn_dynamic/result/` and
refuses to reuse an existing directory. `--output` can name a new directory
only inside that location. Historical results are never overwritten.

Outputs include configuration/environment metadata, dataset SHA-256 and split
indices, training curves as episode CSVs, checkpoint weights, validation tuning
tables, the frozen selection, per-episode evaluation, summaries and paired
intervals. A DQN-owned `result/v4/.gitignore` keeps all generated runs local;
source and documentation remain
reviewable. No commit or push is performed by these scripts.

Do not claim v4 outperforms greedy, v3 or DDQN until adequate repeated runs and
the corresponding comparisons support it. Shared app integration and changes
to the team's reporting scripts require separate approval.
