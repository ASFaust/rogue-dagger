#!/bin/sh
# usage: eval_ckpts.sh FAMILY TAG:CKPT ...   (two deploy settings each: shop 32 tau 1, full table tau 0.25)
# Exits non-zero on the first failure (e.g. CUDA unavailable) instead of silently continuing.
set -e
FAM=$1; shift; P=/shared/work/venv/bin/python
$P -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'eval_ckpts.sh: CUDA not available')"
for spec in "$@"; do
  TAG=${spec%%:*}; CK=${spec#*:}
  $P -m roguedagger.evaluate --method rd --ckpt $CK --family $FAM --device cuda --n-shop 32 --temperature 1.0 --tag $TAG-n32-t1.0 | grep saved
  $P -m roguedagger.evaluate --method rd --ckpt $CK --family $FAM --device cuda --n-shop 0 --temperature 0.25 --tag $TAG-n0-t0.25 | grep saved
done
