#!/bin/sh
# Deployment knobs for one checkpoint on one family: shop size x temperature (+ greedy)
CK=$1; FAM=$2; TAG=$3; P=/shared/work/venv/bin/python
for n in 32 0; do
  for t in 1.0 0.5 0.25; do
    $P -m roguedagger.evaluate --method rd --ckpt $CK --family $FAM --device cuda --n-shop $n --temperature $t --tag $TAG-n$n-t$t | grep saved
  done
  $P -m roguedagger.evaluate --method rd --ckpt $CK --family $FAM --device cuda --n-shop $n --greedy --tag $TAG-n$n-greedy | grep saved
done
