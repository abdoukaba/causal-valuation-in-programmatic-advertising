#!/usr/bin/env bash
set -eu -o pipefail
cd "$(dirname "$0")"
export NO_NUMBA=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ARB_BITS=224
mode=${1:-grid}
case "$mode" in grid|continuous|all) ;; *) echo "Choose grid, continuous or all"; exit 2;; esac
if [[ "$mode" == grid || "$mode" == all ]]; then
    python candidate.py 21 60
    python verify_grid.py 21 60 --power 9 --bits 224
    python verify_grid.py 21 60 --power 28
    python independent_center.py
fi
if [[ "$mode" == all ]]; then
    for grid_size in "11 30" "31 120"; do
        read -r nodes steps <<< "$grid_size"
        python candidate.py "$nodes" "$steps"
    done
    for power in 8 9 12 16; do python verify_grid.py 21 60 --power "$power"; done
    python verify_grid.py 11 30 --power 28
    python verify_grid.py 31 120 --power 28 --bits 160
    python verify_grid.py 31 120 --power 10
    python empirical_check.py
    for eta in .30 .45; do python prior/replay_integer.py "prior/ledger-auction-$eta-40.json.gz" > "results/replay_ipinyou_${eta#.}.json"; done
    (
        cd audit_ipinyou
        python proof/budget_gain.py auction .45 40 1 > results/gain_ipinyou_.45.json
        for method in fem pinn; do python code/validate.py "results/witness_${method}_ipinyou_0.45.json"; done
        python proof/tighten.py .45 ipinyou
        for method in fem pinn; do cmp "results/radius_${method}_ipinyou_0.45.json" "../data/radius_${method}_ipinyou_0.45.json"; done
    )
fi
if [[ "$mode" == continuous || "$mode" == all ]]; then
    (cd audit_ipinyou && python code/validate.py results/witness_fem_composite_0.45.json --bits 224 --parts 24)
    for tolerance in 1000 10000; do python continuum_check.py "$tolerance" 40; done
    if [[ "$mode" == all ]]; then
        for cells in "20 24" "40 48" "80 96"; do
            read -r groups pieces <<< "$cells"
            python dual_residual.py --groups "$groups" --parts "$pieces" --propose
            if [[ "$groups" != 20 ]]; then python dual_residual.py --groups "$groups" --parts "$pieces" --bits 224; fi
        done
    fi
    python dual_residual.py --groups 160 --parts 160 --propose
    if [[ "$mode" == all ]]; then python dual_residual.py --groups 160 --parts 160 --bits 224; fi
    python transfer.py
fi
if [[ "$mode" == all ]]; then python figures.py; fi
python verify_summary.py "$mode"
