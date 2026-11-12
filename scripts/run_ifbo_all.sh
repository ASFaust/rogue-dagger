#!/bin/sh
# FT-PFN + MFPI-random on all three families, CPU workers (single-threaded each)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/shared/work/venv/bin/python
W=${WORKERS:-64}
for fam in taskset pd1 lcbench; do
  $P -m roguedagger.evaluate --method ifbo --family $fam --workers $W || exit 1
done
