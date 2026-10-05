#!/usr/bin/env bash
# Run all experiments. Rendered notebooks are written to output/.
set -euo pipefail

# Shrinkage strength sweep for the calibrated online avg-LR detector.
for k in 0 10 20 30; do
  python run_notebooks.py notebooks/avg_cal_lr.ipynb --alpha "0.023" --k "$k" --save_per_ride # online avg-LR, shrinkage sweep
done

# Operating characteristic sweep across all threshold-based methods.
for a in 0.023 0.05 0.10 0.20; do
  python run_notebooks.py notebooks/avg_cal_lr.ipynb --alpha "$a" --save_per_ride # online avg-LR, per-ride recalibrated threshold
  python run_notebooks.py notebooks/windowed_lr.ipynb --alpha "$a" # per-window LR threshold
  python run_notebooks.py notebooks/batch_lr.ipynb --alpha "$a" # batch LR at end-of-ride
  python run_notebooks.py notebooks/jump_ville_lr.ipynb --alpha "$a" # jumper with Ville threshold
  python run_notebooks.py notebooks/jump_cal_lr.ipynb --alpha "$a" --save_per_ride # jumper with per-ride recalibrated threshold
done

# McNemar test for avg_cal_lr and jump_cal_lr
python run_notebooks.py notebooks/mcnemar_test.ipynb # reads the alpha=0.023 CSVs via notebook defaults

# Collect results to single CSV
python collect_results.py --output_dir output
