#!/bin/sh
# Resume the jobs lost when the container's GPUs dropped out (2026-11-13 ~06:10 UTC).
# Uses GPUs 3-6 (GPU 6 allowed by the user "until further notice"); 0-2 stay free.
# Run from the repo root:  sh scripts/resume_2026-11-13.sh
set -e
cd "$(dirname "$0")/.."
P=/shared/work/venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi > /dev/null || { echo "nvidia-smi fails: GPUs still unavailable"; exit 1; }
$P -c "import torch, sys; sys.exit(0 if torch.cuda.device_count() >= 7 else 'torch sees too few GPUs')"
mkdir -p runs/logs
ev() {  # ev GPU TAG CKPT  -> benchmark evals on all three families, then a synthetic tau sweep
  ( for f in lcbench taskset pd1; do CUDA_VISIBLE_DEVICES=$1 sh scripts/eval_ckpts.sh $f $2:$3; done
    CUDA_VISIBLE_DEVICES=$1 $P scripts/synth_tau.py $3 --n 32 512 --tau 1 0.5 0.25 0.15 0.1 ) > runs/logs/eval-$2.log 2>&1 &
}
ev 3 wlong300k runs/wide-long-v1/last.pt
ev 4 wbig100k  runs/wide-bigshop-v1/last.pt
ev 5 wt025     runs/wide-t025-v1/last.pt
ev 6 wread100k runs/wide-readout-v1/last.pt
# the pair-weighting 0.3 training run that crashed at start, next to the readout eval on GPU 6
CUDA_VISIBLE_DEVICES=6 nohup $P -m roguedagger.train --config configs/wide_focus03.yaml --set name=wide-focus03-v1 \
  > runs/logs/wide-focus03-v1.log 2>&1 &
echo "started; logs in runs/logs/eval-{wlong300k,wbig100k,wt025,wread100k}.log and runs/logs/wide-focus03-v1.log"
