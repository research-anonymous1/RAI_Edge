#!/usr/bin/env bash
# Runs every experiment of the paper. Call from the code/ directory:  bash ../run_all.sh
set -e
W=${WORKERS:-2}
for e in main series calib ablation beta alpha alpha2 horizon gm burst burst2 capacity spread backhaul; do
  echo "=== $e"; python run_experiments.py "$e" --workers "$W"
done
