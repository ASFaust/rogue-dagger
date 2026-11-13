#!/bin/sh
# usage: [T32=1.0] [TFULL=0.25] eval_ckpts.sh FAMILY TAG:CKPT ...
# Two deploy settings each: shop 32 at temperature T32, full table at TFULL (RL checkpoints: use rl.temperature).
# Exits non-zero on the first failure (e.g. CUDA unavailable) instead of silently continuing.
set -e
FAM=$1; shift; P=/shared/work/venv/bin/python
T32=${T32:-1.0}; TFULL=${TFULL:-0.25}
$P -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'eval_ckpts.sh: CUDA not available')"
for spec in "$@"; do
  TAG=${spec%%:*}; CK=${spec#*:}
  $P -m roguedagger.evaluate --method rd --ckpt $CK --family $FAM --device cuda --n-shop 32 --temperature $T32 --tag $TAG-n32-t$T32 | grep saved
  $P -m roguedagger.evaluate --method rd --ckpt $CK --family $FAM --device cuda --n-shop 0 --temperature $TFULL --tag $TAG-n0-t$TFULL | grep saved
done
