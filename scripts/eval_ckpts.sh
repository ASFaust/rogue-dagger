#!/bin/sh
# usage: eval_ckpts.sh FAMILY TAG:CKPT ...   (two deploy settings each: shop 32 tau 1, full table tau 0.25)
FAM=$1; shift; P=/shared/work/venv/bin/python
for spec in "$@"; do
  TAG=${spec%%:*}; CK=${spec#*:}
  $P -m roguedagger.evaluate --method rd --ckpt $CK --family $FAM --device cuda --n-shop 32 --temperature 1.0 --tag $TAG-n32-t1.0 | grep saved
  $P -m roguedagger.evaluate --method rd --ckpt $CK --family $FAM --device cuda --n-shop 0 --temperature 0.25 --tag $TAG-n0-t0.25 | grep saved
done
