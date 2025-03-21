import numpy as np
import mlx.core as mx
from mlx.core.fft import ifft
import time
import multiprocessing
import matplotlib.pyplot as plt

def define_match_filter(a, b, stream): # multiplies two arrays, performs the IFFT, and returns the maximum value of the reshaped result.
    product = mx.multiply(a.conj(), b, stream=stream)
    ifft_result = ifft(product, stream=stream)
    reshaped_result = ifft_result.reshape(ifft_result.size) # reshapes the IFFT result
    return mx.eval(mx.max(reshaped_result, stream=stream)) # returns max val

def frozen_cpumatchfilter(ab):
                return define_match_filter(*ab, stream=mx.cpu)

def frozen_gpumatchfilter(ab):
                return define_match_filter(*ab, stream=mx.gpu)

if __name__ == "__main__":
    list_size = 120
    array_size = int(2**20)
    a_list = [mx.array(mx.random.normal(shape=(array_size,), dtype=mx.float32) + 1j * mx.random.normal(shape=(array_size,), dtype=mx.float32)) for _ in range(list_size)]
    b_list = [mx.array(mx.random.normal(shape=(array_size,), dtype=mx.float32) + 1j * mx.random.normal(shape=(array_size,), dtype=mx.float32)) for _ in range(list_size)]
    ab_list = list(zip(a_list,b_list))
    cpu_max_values = []
    gpu_max_values = []

    # running on a single core for CPU and GPU
    for a, b in zip(a_list, b_list):
        cpu_max_values.append(define_match_filter(a, b, stream=mx.cpu))
        gpu_max_values.append(define_match_filter(a, b, stream=mx.gpu))

    # parallelizing the task (cores 1-4)
    cpu_times_parallel = []
    gpu_times_parallel = []

    test_cores = [1,2,3]

    for ncores in test_cores:
        with multiprocessing.Pool(ncores) as pool:
            start_time = time.perf_counter()
            cpu_results = list(pool.map(frozen_cpumatchfilter, ab_list))
            cpu_time_parallel = time.perf_counter() - start_time
            cpu_times_parallel.append(cpu_time_parallel)

            start_time = time.perf_counter()
            gpu_results = list(pool.map(frozen_gpumatchfilter, ab_list))
            gpu_time_parallel = time.perf_counter() - start_time
            gpu_times_parallel.append(gpu_time_parallel)

    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.plot(test_cores, cpu_times_parallel, label="CPU (Parallel)", marker='o')
    plt.plot(test_cores, gpu_times_parallel, label="GPU (Parallel)", marker='x')
    plt.xlabel("Number of Cores")
    plt.ylabel("Time (seconds)")
    plt.title("Performance of Match Filter on CPU vs GPU (Parallelized)")
    plt.legend()
    plt.grid(True)

    # saving plot
    plt.savefig("match_filter_performance.png")
    plt.show()
