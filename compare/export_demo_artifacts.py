"""Export real, validation-selected Part A checkpoints for the dashboard.

Run after a completed ``compare.part_a_comparison`` experiment. The export
contains only checkpoints that actually exist; it never invents early stages.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "compare" / "checkpoints"
MODELS = ("Q-Learning", "DQN", "Double DQN", "Dueling DQN", "D3QN")


def checkpoint_name(model, seed, episode=None):
    stem = model.replace(" ", "_")
    suffix = ".pkl" if model == "Q-Learning" else ".pt"
    if episode is not None:
        return f"{stem}_seed{seed}_ep{episode}{suffix}"
    return f"{stem}_seed{seed}{suffix}"


def rank(row):
    """Validation success, then on-time rate, then shared-environment reward."""
    return (float(row["delivered"]), float(row["on_time"]), float(row["reward"]))


def export(run_dir, dest=DEFAULT_DEST):
    run_dir, dest = Path(run_dir), Path(dest)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("smoke"):
        raise ValueError("Only a completed non-smoke run can be exported")
    with (run_dir / "learning.csv").open(newline="", encoding="utf-8") as handle:
        learning = list(csv.DictReader(handle))
    with (run_dir / "validation_per_seed.csv").open(newline="", encoding="utf-8") as handle:
        final_validation = list(csv.DictReader(handle))
    with (run_dir / "validation_episodes.csv").open(newline="", encoding="utf-8") as handle:
        validation_episodes = list(csv.DictReader(handle))
    by_seed = {(row["model"], int(row["training_seed"])): row for row in final_validation}
    last_log = {}
    for row in learning:
        last_log[(row["model"], int(row["training_seed"]))] = row

    stage_file = run_dir / "stage_validation.csv"
    # The stage file is optional for historical runs. Only existing weights count.
    if stage_file.exists():
        with stage_file.open(newline="", encoding="utf-8") as handle:
            stage_rows = list(csv.DictReader(handle))
    else:
        stage_rows = []

    dest.mkdir(parents=True, exist_ok=True)
    catalog = {
        "source_run": run_dir.name,
        "data_sha256": manifest["data_sha256"],
        "selection": "validation delivered, then on-time, then common reward",
        "train_steps": int(manifest["settings"]["train_steps"]),
        "training_seeds": manifest["seeds"],
        "validation_trips_per_checkpoint": int(manifest["eval_episodes"]),
        "train_map_seed": manifest["train_map_seed"],
        "validation_map_seed": manifest["validation_map_seed"],
        "models": {},
    }
    selected_logs, validation_progress = [], []
    for model in MODELS:
        candidates = [row for row in final_validation if row["model"] == model]
        if not candidates:
            continue
        all_candidates = list(candidates)
        all_candidates.extend(row for row in stage_rows if row["model"] == model
                              and (run_dir / row["file"]).is_file())
        best = max(all_candidates, key=rank)
        seed = int(best["training_seed"])
        best_final = by_seed[(model, seed)]
        last = last_log[(model, seed)]
        final_episode = int(last["episodes"])
        final_source = run_dir / checkpoint_name(model, seed)
        if not final_source.is_file():
            continue
        final_name = checkpoint_name(model, seed)
        shutil.copy2(final_source, dest / final_name)
        selected_logs.extend(row for row in learning
                             if row["model"] == model and int(row["training_seed"]) == seed)
        completed_final = [float(row["delivery_min"]) for row in validation_episodes
                           if row["model"] == model and int(row["training_seed"]) == seed
                           and row["delivered"] == "1" and row["delivery_min"]]
        final_delivery_min = sum(completed_final) / len(completed_final) if completed_final else ""
        validation_progress.append(dict(model=model, training_seed=seed,
                                        episode=final_episode, step=catalog["train_steps"],
                                        delivered=best_final["delivered"],
                                        on_time=best_final["on_time"], reward=best_final["reward"],
                                        delivery_min=final_delivery_min, steps=best_final["steps"],
                                        checkpoint="final"))
        stages = [{"label": "Latest (final) checkpoint", "episode": final_episode,
                   "file": final_name, "training_seed": seed, "kind": "final"}]
        stage_candidates = [row for row in stage_rows if row["model"] == model
                            and int(row["training_seed"]) == seed
                            and (run_dir / row["file"]).is_file()]
        for row in stage_candidates:
            source = run_dir / row["file"]
            shutil.copy2(source, dest / source.name)
            episode = int(row["episode"])
            stages.append({"label": "Before training" if episode == 0 else f"After {episode} episodes",
                           "episode": episode, "file": source.name,
                           "training_seed": int(row["training_seed"]), "kind": "milestone"})
            validation_progress.append(dict(model=model, training_seed=seed,
                                            episode=episode, step=row["step"],
                                            delivered=row["delivered"], on_time=row["on_time"],
                                            reward=row["reward"],
                                            delivery_min=row.get("delivery_min", ""),
                                            steps=row["steps"], checkpoint="milestone"))
        if "file" in best:
            best_file = Path(best["file"]).name
            best_episode = int(best["episode"])
            best_seed = int(best["training_seed"])
        else:
            best_file, best_episode, best_seed = final_name, final_episode, seed
        stages.insert(0, {"label": "Best validated checkpoint", "episode": best_episode,
                          "file": best_file, "training_seed": best_seed, "kind": "best"})
        catalog["models"][model] = {
            "selected_seed": seed,
            "total_completed_episodes": final_episode,
            "validation": {key: float(best_final[key]) for key in
                           ("delivered", "on_time", "reward", "elapsed_min", "steps")},
            "best_validation": {key: float(best[key]) for key in
                                ("delivered", "on_time", "reward", "elapsed_min", "steps")},
            "stages": stages,
        }

    fields = ("model", "training_seed", "step", "episodes", "mean_recent_reward",
              "recent_success", "updates")
    with (dest / "learning.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in selected_logs)
    progress_fields = ("model", "training_seed", "episode", "step", "delivered",
                       "on_time", "reward", "delivery_min", "steps", "checkpoint")
    with (dest / "validation_progress.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=progress_fields)
        writer.writeheader()
        writer.writerows(validation_progress)
    all_progress = []
    for row in final_validation:
        model, seed = row["model"], int(row["training_seed"])
        completed = [float(item["delivery_min"]) for item in validation_episodes
                     if item["model"] == model and int(item["training_seed"]) == seed
                     and item["delivered"] == "1" and item["delivery_min"]]
        all_progress.append(dict(model=model, training_seed=seed,
                                 episode=int(last_log[(model, seed)]["episodes"]),
                                 step=catalog["train_steps"], delivered=row["delivered"],
                                 on_time=row["on_time"], reward=row["reward"],
                                 delivery_min=sum(completed) / len(completed) if completed else "",
                                 steps=row["steps"], checkpoint="final"))
    for row in stage_rows:
        if not (run_dir / row["file"]).is_file():
            continue
        all_progress.append(dict(model=row["model"], training_seed=row["training_seed"],
                                 episode=row["episode"], step=row["step"],
                                 delivered=row["delivered"], on_time=row["on_time"],
                                 reward=row["reward"],
                                 delivery_min=row.get("delivery_min", ""),
                                 steps=row["steps"], checkpoint="milestone"))
    with (dest / "all_validation_progress.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=progress_fields)
        writer.writeheader()
        writer.writerows(all_progress)
    (dest / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(list(export(args.run_dir)["models"])))
