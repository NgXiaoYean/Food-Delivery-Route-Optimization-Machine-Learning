# DQN v4 implementation and verification

## Scope

Implemented on `shinnie`, entirely inside `dqn/`. Shared environment/evaluation
files, the application, Q-Learning, DDQN, and original DQN v1/v2/v3 sources and
artifacts were preserved. No commit or push was performed.

The contribution adds a standard-DQN training pipeline with synthetic traffic
and weather events during trips, deadline-aware/local-cost features, a direct
travel-time reward, finite-task terminal handling, train/validation/test
separation, repeated training seeds, component comparisons and paired metrics.

See [README_V4.md](README_V4.md) for commands and simulator assumptions.

## Verification

- 20 behavioral tests passed, including seeded event reproducibility,
  action-independent conditions, deadline features, terminal targets, split
  separation, paired comparisons and checkpoint round trips.
- The end-to-end smoke experiment completed all training/evaluation stages.
- A 5,000-step, three-seed validation-only pilot completed. It was insufficient
  to establish useful routing performance and is not evidence of superiority.
- `python -B dqn_dynamic/part_b_refine.py --help` confirmed that the original Part B
  command now exposes the new runner.
- A Windows console encoding issue found during the first smoke attempt was
  fixed by displaying the output path relative to the project.

## Repeated-seed benchmark

Run directory: `dqn_dynamic/result/20261115T033512521330Z/`.

```powershell
python -B -m dqn_dynamic.run_experiments --steps 60000 --seeds 11 22 33 --eval-episodes 200
```

All training runs receive 60,000 environment steps and 14,751 optimizer
updates. Configurations change one parameter at a time. The validation
selection uses average delivery success, breaking ties with common return.

| Configuration | Validation success | Validation on-time rate | Mean common return |
| --- | ---: | ---: | ---: |
| Starting configuration | 52.17% | 22.00% | -96.44 |
| Lower learning rate (0.0005) | 27.33% | 8.83% | -154.80 |
| Lower discount (0.95) | **78.83%** | **32.50%** | **-40.44** |
| Faster target updates (250 steps) | 46.50% | 20.83% | -109.16 |

The selected configuration is standard DQN with learning rate 0.001, discount
0.95, 128 hidden units, and target updates every 500 environment steps.
Selection was recorded before final test evaluation. The component comparisons
use that fixed configuration; they do not independently retune each variant.

The full run completed: **24 training runs, 1,440,000 training interactions and
11,400 final evaluation episodes**. There are 200 matched episodes per scenario
for each policy/training-seed combination. Baselines and ensembles run once per
scenario. None of the test outcomes were used to change the configuration.

### Dynamic test conditions

Values for learned variants are means over all three seeds. Duration is the
mean of each seed's completed-delivery mean; always read it beside success.

| Variant | Delivered | On time | Delivery minutes | Common return |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 100.00% | 46.00% | 29.71 | 16.18 |
| Current-map time planner | 100.00% | 57.00% | 25.72 | 21.61 |
| v3-style dynamic control (`baseline`) | 100.00% | 56.17% | 26.04 | 21.18 |
| Deadline state + legacy reward (`state_only`) | 100.00% | 56.17% | 26.05 | 21.18 |
| Original-style state + time reward (`reward_only`) | 100.00% | 54.33% | 26.55 | 20.53 |
| Combined dynamic model (`full_dynamic`) | 77.17% | 39.33% | 39.73 | -39.86 |
| Combined static-training control (`full_static`) | 100.00% | 56.50% | 26.02 | 21.23 |
| Equal-weight full-model ensemble | 97.00% | 50.00% | 29.05 | 10.33 |
| Ensemble + backtracking penalty | 99.50% | 52.50% | 27.70 | 17.61 |

The `baseline` above is freshly trained on the new protocol, **not** the old
saved v3 checkpoint. These numbers are not directly comparable to the team's
historical static-environment scoreboard.

### Static and stress checks

| Scenario | Variant | Delivered | On time | Delivery minutes | Common return |
| --- | --- | ---: | ---: | ---: | ---: |
| Static | Greedy | 100.00% | 51.50% | 28.28 | 18.11 |
| Static | v3-style dynamic control | 100.00% | 56.67% | 26.02 | 21.20 |
| Static | Deadline state + legacy reward | 100.00% | 57.33% | 25.95 | 21.29 |
| Static | Combined dynamic model | 74.83% | 39.67% | 38.62 | -43.19 |
| Stress | Greedy | 100.00% | 40.50% | 33.50 | 10.94 |
| Stress | v3-style dynamic control | 100.00% | 48.50% | 29.52 | 16.61 |
| Stress | Deadline state + legacy reward | 100.00% | 47.00% | 29.68 | 16.40 |
| Stress | Combined dynamic model | 81.50% | 28.83% | 50.53 | -44.68 |

### What the experiment supports

- Standard DQN can route in the new changing-condition simulator and outperform
  greedy without Double/Dueling DQN. The dynamic control improves common return
  by **5.01**, with an exploratory paired 95% interval **[4.22, 5.83]**.
- The deadline-state/legacy-reward variant also improves common return over
  greedy by **5.00**, interval **[4.19, 5.84]**, and uses about 12.3% less delivery
  time. It does not show a meaningful incremental advantage over the simpler
  DQN control in this run.
- The full combination is unstable. Its three dynamic-test success rates are
  **91.5%, 40.5%, and 99.5%**. Reporting only the strongest seed would hide the
  problem. Its common-return difference versus the dynamic control is -61.04,
  interval **[-153.79, -2.32]**.
- Dynamic training is not demonstrated to improve this full model: its otherwise
  matched static-training control generalizes better in this experiment.
- The backtracking heuristic helps the full-model ensemble, but the ensemble
  still does not surpass the simpler individual DQN variants in these results.
- 91.5% of greedy dynamic-test trips crossed at least one scheduled event; short
  trips can finish before changes. The full tables retain event exposure counts.

The functionality is implemented and tested, but the combined model should not
be treated as a proven improvement or promoted over existing models. Further
work should use validation data to investigate its training instability (for
example, a static-to-dynamic curriculum or controlled reward-coefficient and
training-budget studies). Feature/variant selection must be validated before
any new confirmatory test; these test results are already observed.

Detailed evidence in the run directory:

- `manifest.json`: complete status, dataset/split information and runtime settings.
- `selection.json`: validation-selected hyperparameters and fixed ensemble setup.
- `validation_tuning.csv`: all candidate/seed validation results.
- `test_summary.csv`: per-seed results for every scenario and policy.
- `test_episodes.csv`: all 11,400 episode records.
- `test_paired_intervals.csv`: differences versus greedy and learned controls.

## Interpretation limits

More features and dynamic conditions do not guarantee improved performance.
Validation success of 78.83% is insufficient to claim delivery reliability
comparable to the direct-route baseline. Report failures, not just travel times
among successful deliveries.

Historical dataset duration is a deadline proxy, not a promised SLA. Traffic
events are synthetic; travel costs are sampled at move departure. The DQN has
local observations, while the time planner sees the complete current map.

The final order partition preserves the team's historical test rows, which
were inspected in older experiments. It is excluded from new tuning but is
not a previously untouched dataset. Test maps/events are held out from the
new training/validation banks. Three-seed bootstrap intervals are exploratory.

Generated checkpoints and detailed tables remain local under `dqn_dynamic/result/`
and are ignored by its DQN-owned `.gitignore`. The source and this report can
be reviewed separately from those large artifacts.
