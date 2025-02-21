import multiprocessing
import time
import matplotlib.pyplot as plt

def multiply_pair(pair):
    return pair[0] * pair[1]

def run_multiprocessing(num_cores, data):
    with multiprocessing.Pool(processes=num_cores) as pool:
        start_time = time.perf_counter()
        results = pool.map(multiply_pair, data) # use a named function
        end_time = time.perf_counter()
    
    return end_time - start_time, results

if __name__ == '__main__':
    num_cores_list = [1, 2, 4, 8] 
    times = []

    data = [(i, i+1) for i in range(1000000)]  # large dataset example

    for num_cores in num_cores_list:
        print(f"Testing with {num_cores} core(s)")
        elapsed_time, _ = run_multiprocessing(num_cores, data)
        times.append(elapsed_time)
        print(f"Time for {num_cores} core(s): {elapsed_time:.4f} seconds")

    plt.figure(figsize=(10, 6))
    plt.plot(num_cores_list, times, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Cores')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Number of Cores for Multiplication Task')
    plt.grid(True)
    plt.show()
