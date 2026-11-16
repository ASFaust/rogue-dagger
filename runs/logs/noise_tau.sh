#!/bin/sh
# Frozen. Deploy-temperature check (user, 2026-11-16): does the noise prior's loss come from tau 0.25 being tuned for
# the noisy prior? Full table at tau 0.5 and 1.0 for noise and focus03 seeds alike (GPUs 3, 4, 5).
cd /shared/work/private/rogue-dagger; export CUDA_DEVICE_ORDER=PCI_BUS_ID; P=/shared/work/venv/bin/python
one() { # GPU TAG CKPT TAG CKPT
  g=$1; shift
  while [ $# -gt 0 ]; do
    for t in 0.5 1.0; do for fam in taskset pd1 lcbench; do
      CUDA_VISIBLE_DEVICES=$g $P -m roguedagger.evaluate --method rd --ckpt $2 --family $fam --device cuda --n-shop 0 --temperature $t --tag $1-n0-t$t | grep saved
    done; done; shift 2
  done
}
one 3 wf03noise runs/wide-focus03-noise-v1/last.pt wfocus03 runs/wide-focus03-v1/last.pt > runs/logs/noise_tau_g3.log 2>&1 &
one 4 wf03noises1 runs/wide-focus03-noise-s1/last.pt wf03s1 runs/wide-focus03-s1/last.pt > runs/logs/noise_tau_g4.log 2>&1 &
one 5 wf03noises2 runs/wide-focus03-noise-s2/last.pt wf03s2 runs/wide-focus03-s2/last.pt > runs/logs/noise_tau_g5.log 2>&1 &
wait
