# Experiment protocol (one page)

Every result in the report comes from ONE of the scripts below. Anything else is historical.

| Report part | Task | Script | Split | Seeds | Main metrics |
|---|---|---|---|---|---|
| Part A: choose the algorithm | Single order, `env.py` | `python -m compare.part_a_comparison` | train/val/test (3-way, see `dqn/experiments.py`); choice made on validation only | 5 (11,22,33,44,55) | delivered %, on-time %, trip minutes, reward |
| Part A cross-check: why D3QN | Multi order, `env_multi.py` | `python d3qn/ladder_multi.py --seeds 3` | 70/15/15, validation seeds 1000+ | 3 | reward, paired bootstrap 95% CI |
| Part B: tune + final model | Multi order | `python d3qn/d3qn_multi.py --stage A/B/C` | same 70/15/15 | 3+ | reward vs Greedy and Planner |
| Part B: ensemble, ablation, robustness, retrain, errors, timing | Multi order | `d3qn/ensemble_multi.py`, `ablation_multi.py`, `robustness_multi.py`, `retrain_study_multi.py`, `error_analysis_multi.py`, `timing_multi.py` | test set (seeds 0+) | see script | reward, gap closed |
| Split sensitivity | Multi order | `python d3qn/split_study_multi.py --seeds 3` | 5 split schemes | 3 | ensemble gain over Greedy |
| Extension (not the main story) | Single order, dynamic traffic, standard DQN only | `python -B -m dqn_dynamic.run_experiments` | own split | 3 | see `dqn_dynamic/README_V4.md` |

Rules: (1) choose on validation, report on test; (2) same trips for every model in a comparison;
(3) always report seeds and a CI or std; (4) say so when a split was reused for tuning.
Historical files: `dqn/dqn_v1..v3.py` and `d3qn/d3qn_pipeline.py` (the original single-order models the app loads).
Plain name rule: D3QN = Dueling Double DQN.
