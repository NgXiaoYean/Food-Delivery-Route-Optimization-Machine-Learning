# Dashboard learning checkpoints

These exported checkpoints come from the completed, non-smoke five-model
Part A run named in `catalog.json`. For each algorithm, the export chooses
the best **available** checkpoint across five training seeds and genuine
episode milestones using validation delivery rate, then validation on-time
rate, then validation reward. `all_validation_progress.csv` records the
selection evidence. The accompanying `learning.csv` and
`validation_progress.csv` contain the selected seed's training and
validation curves. The best validated checkpoint may equal the latest
checkpoint; the dashboard labels both honestly. For final checkpoints,
the episode number is the number of completed episodes at the 60,000-step
cutoff; training may have entered another episode before saving.

To reproduce the genuine early-stage routes and validation curves, install
the project requirements and rerun from the repository root:

```powershell
python -B -m compare.part_a_comparison --record-stages
```

The run saves initial weights, the requested completed-episode milestones,
and validation results on the same scenarios for each model. It then exports
the validation-selected checkpoints and progress data here. This is a new
experiment and may take substantially longer than the original comparison.
Do not describe the training-only progress curves as held-out validation.

