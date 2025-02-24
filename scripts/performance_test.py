import mlx.core as mx
import time
import matplotlib.pyplot as plt
import argparse
import os

def run_performance_test(output_file="performance_plot.png"):
    sample_rate = 4096  # 4096 samples per second
    duration = 10  # 10 seconds of data
    total_samples = sample_rate * duration  # total data points
    iterations = 50 

    data = mx.random.normal(shape=(total_samples,), dtype=mx.float32)  # generate random signal data
    
    reshaped_data = mx.reshape(data, (duration, sample_rate))  # reshape into (duration, 4096)

    cpu_times = []
    gpu_times = []

    # local max per second (CPU)
    cpu_time = 0
    for _ in range(iterations):
        start_time = time.perf_counter()
        mx.eval(mx.argmax(reshaped_data, axis=1, stream=mx.cpu))
        cpu_time += time.perf_counter() - start_time
    cpu_times.append(cpu_time / iterations)

    # local max per second (GPU)
    gpu_time = 0
    for _ in range(iterations):
        start_time = time.perf_counter()
        mx.eval(mx.argmax(reshaped_data, axis=1, stream=mx.gpu))
        gpu_time += time.perf_counter() - start_time
    gpu_times.append(gpu_time / iterations)

    plt.figure(figsize=(10, 6))
    plt.plot(["CPU"], [cpu_times[0]], marker='o', linestyle='-', color="blue", label="CPU")
    plt.plot(["GPU"], [gpu_times[0]], marker='o', linestyle='-', color="red", label="GPU")
    plt.ylabel("Time (seconds)")
    plt.title("Performance of Local Maximum Detection (CPU vs GPU)")
    plt.grid(True)

    # save plot
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

def load_plot(image_file="performance_plot.png"):
    """ load and display saved plot """
    if not os.path.exists(image_file):
        print(f"run script first to generate plot")
        return

    img = plt.imread(image_file)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Saved Performance Plot")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance Test Script")
    parser.add_argument("--load", action="store_true", help="load and display the saved plot instead of running the test")
    parser.add_argument("--output", type=str, default="performance_plot.png", help="filename that saves/loads the plot")

    args = parser.parse_args()

    if args.load:
        load_plot(args.output)
    else:
        run_performance_test(args.output)
