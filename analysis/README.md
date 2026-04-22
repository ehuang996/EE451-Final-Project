# Analysis Layout

This directory is organized by run:

- [run1/](run1/) — archived figures from the original pure-C++ sweep
- [run2/](run2/) — refreshed figures from the BLAS-backed rerun

The shared plotting script is [plot_sweep.py](plot_sweep.py). It reads
`results/run2/` and writes refreshed figures to `analysis/run2/figures/` and
`EE451-Final-Paper/figures/`.
