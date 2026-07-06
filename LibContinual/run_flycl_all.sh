#!/bin/bash
# Reproduce Fly-CL inside LibContinual end-to-end on all three datasets (GPU).
# Prereqs: assets/vit_b16_augreg_in21k.npz present, data/ points at the datasets,
# deps installed (see README §3.1). Reads Ā from '[Batch] Overall Avg Acc'.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
LOGDIR=${LOGDIR:-./results_logs}
mkdir -p "$LOGDIR"

for cfg in flycl flycl_cub flycl_vtab; do
  echo "########## START $cfg  $(date) ##########"
  python run_trainer.py --config "$cfg" > "$LOGDIR/${cfg}.log" 2>&1
  echo "########## DONE  $cfg  rc=$?  $(date) ##########"
  grep -E "Batch\] Overall Avg Acc|Time Costs" "$LOGDIR/${cfg}.log" | tail -2
done
echo "ALL_DONE $(date)"
