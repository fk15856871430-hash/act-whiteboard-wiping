#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/home/user/whiteboard_wipe_20260916_17_rotvec_v3_clean}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train/act_whiteboard_rotvec_v3_optimized}"

if [[ ! -f "${DATASET_ROOT}/meta/whiteboard_preparation.json" ]]; then
  echo "Prepared dataset not found: ${DATASET_ROOT}" >&2
  echo "Run: conda run -n lerobot python ${REPO_ROOT}/scripts/prepare_whiteboard_act_dataset.py" >&2
  exit 1
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output already exists; set OUTPUT_DIR to a new path: ${OUTPUT_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

lerobot-train \
  --dataset.repo_id=local/whiteboard_wipe_rotvec_clean \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.eval_split=0.1 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.85,1.15]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.85,1.15]}},"saturation":{"weight":0.5,"type":"ColorJitter","kwargs":{"saturation":[0.85,1.15]}},"sharpness":{"weight":0.5,"type":"SharpnessJitter","kwargs":{"sharpness":[0.8,1.2]}}}' \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.use_vae=false \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --scheduler.type=cosine_annealing_with_warmup \
  --scheduler.num_warmup_steps=1000 \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps=8 \
  --batch_size=8 \
  --num_workers=8 \
  --log_freq=100 \
  --steps=120000 \
  --eval_steps=5000 \
  --max_eval_samples=4096 \
  --env_eval_freq=0 \
  --save_freq=5000 \
  --ema.enable=true \
  --ema.decay=0.999 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_whiteboard_rotvec_v3_optimized \
  --wandb.enable=true \
  --wandb.mode=offline
