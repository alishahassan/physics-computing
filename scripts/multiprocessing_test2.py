import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt

def multiply_pair(pair):
    return pair[0] * pair[1]

def run_multiprocessing(num_cores, data):
    with multiprocessing.Pool(processes=num_cores) as pool:
        start_time = time.perf_counter()
        results = pool.map(multiply_pair, data)  # distribute workload using map
        end_time = time.perf_counter()
    
    return end_time - start_time, results  # return execution time and results

if __name__ == '__main__':
    num_cores_list = [1, 2, 4, 8]  # number of cores to test
    times = []

    size = 1000000  # large dataset size
    array1 = np.arange(size)  # create an array: [0, 1, 2, ..., 999999]
    array2 = np.arange(size, 2 * size)  # create another array: [1000000, 1000001, ..., 1999999]

    # create a list of pairs (tuples) for multiplication
    data = list(zip(array1, array2))

    for num_cores in num_cores_list:
        print(f"Testing with {num_cores} core(s)")
        elapsed_time, _ = run_multiprocessing(num_cores, data)
        times.append(elapsed_time)
        print(f"Time for {num_cores} core(s): {elapsed_time:.4f} seconds")

    # plot execution time vs number of cores
    plt.figure(figsize=(10, 6))
    plt.plot(num_cores_list, times, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Cores')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Number of Cores')
    plt.grid(True)
    plt.show()
