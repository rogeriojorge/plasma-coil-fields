# Source from the repository root after setting DEPS to the directory holding the pinned checkouts.
export PYTHONPATH=$DEPS/vmex_pinned:$DEPS/wt_essos_pc:$DEPS/wt_pyqsc_ro/src:$DEPS/wt_solvax027/src:$DEPS/pydeps
export JAX_ENABLE_X64=1 OMP_NUM_THREADS=4
