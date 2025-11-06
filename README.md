# Physics Computing

- Contains benchmarking and computational scripts used in astrophysics-related research.  
- Focuses on analyzing CPU vs GPU performance, parallelization efficiency, and match filter operations for signal processing.

## Contents

- **matchfilter_benchmark.py**: Runs benchmark tests comparing CPU and GPU performance across different core counts and saves results to an HDF5 file.
- **matchfilter_plot.py**: Reads benchmark results and generates comparative plots (CPU vs GPU performance, speedup factors).
- **multiprocessing tests**: Experiments using Python’s multiprocessing to evaluate how execution time scales with different numbers of cores.
- **utility scripts**: Contain helper functions and examples for performing numerical operations in parallel or with partial functions.

## Purpose

These scripts were developed as part of astrophysics research to test and visualize computational performance for physics simulations and signal analysis tasks.  
They serve as a foundation for exploring performance trade-offs in high-compute workflows involving FFTs, parallel processing, and GPU acceleration.
