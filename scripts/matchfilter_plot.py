# how to run plotting:
#   python matchfilter_plot.py --input match_filter_results.h5

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_results(input_file):
    with h5py.File(input_file, "r") as hdf:
        test_cores = hdf["test_cores"][:]
        cpu_times_parallel = hdf.get("cpu_times_parallel", None)
        gpu_times_parallel = hdf.get("gpu_times_parallel", None)

        if cpu_times_parallel is not None:
            cpu_times_parallel = hdf["cpu_times_parallel"][:]
        if gpu_times_parallel is not None:
            gpu_times_parallel = hdf["gpu_times_parallel"][:]

    plt.figure(figsize=(10, 6))

# only plots if CPU and GPU times are available

    if cpu_times_parallel is not None:
        plt.plot(test_cores, cpu_times_parallel, label="CPU (Parallel)", marker='o')
    if gpu_times_parallel is not None:
        plt.plot(test_cores, gpu_times_parallel, label="GPU (Parallel)", marker='x')

    plt.xlabel("Number of Cores")
    plt.ylabel("Time (per Match Filter in Seconds)")
    plt.title("Performance of Match Filter on CPU vs GPU (Parallelized)")
    plt.legend()
    plt.grid(True)
    plt.savefig("match_filter_performance.png")
    plt.show()

    if cpu_times_parallel is not None and gpu_times_parallel is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(test_cores, cpu_times_parallel / gpu_times_parallel, label="Speedup Factor (CPU/GPU)", marker='o')
        plt.xlabel("Number of Cores")
        plt.ylabel("CPU Time / GPU Time")
        plt.title("Speed Up Factor")
        plt.legend()
        plt.grid(True)
        plt.savefig("match_filter_speedupplot.png")
        plt.show()

# reads CPU & GPU benchmark results
# plots: time per match filter (CPU vs GPU), Speedup (CPU time divided by GPU time)
#        save plot as png files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot match filter benchmark results.")
    parser.add_argument("--input", type=str, default="match_filter_results.h5", help="HDF5 file with benchmark results")
    args = parser.parse_args()

    plot_results(args.input)
