#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/workspace
POCR_DIR=${WORKDIR}/PaddleOCR

# 1) Get PaddleOCR repo (only once)
if [ ! -d "$POCR_DIR" ]; then
  git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git "$POCR_DIR"
fi

# 2) Python deps for training
python3 -m pip install -r ${POCR_DIR}/requirements.txt

# 3) Dataset paths (you create these)
# Expected:
#   /workspace/data/rec_dataset/train/images/*.jpg
#   /workspace/data/rec_dataset/train/rec_gt.txt
#   /workspace/data/rec_dataset/val/images/*.jpg
#   /workspace/data/rec_dataset/val/rec_gt.txt
export REC_ROOT=/workspace/data/rec_dataset

# 4) Choose a baseline config (PP-OCRv4 recognition, English)
CFG=${POCR_DIR}/configs/rec/PP-OCRv4/rec_en_PP-OCRv4_distillation.yml

# 5) Override paths on the fly via env (PaddleOCR supports this)
python3 ${POCR_DIR}/tools/train.py \
  -c ${CFG} \
  -o Global.use_gpu=True \
     Global.save_model_dir=/workspace/models/rec_ppocrv4 \
     Global.save_epoch_step=1 \
     Global.eval_batch_step=200 \
     Train.dataset.data_dir=${REC_ROOT}/train/images \
     Train.loader.batch_size_per_card=256 \
     Train.dataset.label_file_list=[\"${REC_ROOT}/train/rec_gt.txt\"] \
     Eval.dataset.data_dir=${REC_ROOT}/val/images \
     Eval.dataset.label_file_list=[\"${REC_ROOT}/val/rec_gt.txt\"]