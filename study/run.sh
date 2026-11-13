#!/usr/bin/env bash
# Usage: study/run.sh refs [--jobs N --cpus a,b,..]   |   study/run.sh time [--out F --filter RE --solvers ...]
# Timing runs pinned to one logical CPU (CPU=4 by default; keep its hyperthread sibling idle) with
# single-threaded BLAS unless OPENBLAS_NUM_THREADS is set by the caller. MPSolve (optional, drop it with
# --solvers numpy,numpy_eig,numpy_newton,mas) is loaded from the system or from $MPSOLVE_LIBDIR.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
if [ "$1" = time ]; then
    if [ -n "${MPSOLVE_LIBDIR:-}" ]; then
        export LD_LIBRARY_PATH=$MPSOLVE_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        [ -e "$MPSOLVE_LIBDIR/libtcmalloc_minimal.so.4" ] && export LD_PRELOAD=$MPSOLVE_LIBDIR/libtcmalloc_minimal.so.4
    fi
    exec taskset -c "${CPU:-4}" python3 run_study.py "$@"
fi
exec python3 run_study.py "$@"
