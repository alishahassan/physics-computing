import mlx.core as mx
import numpy as np

def fun(a, b, d1, d2):
  x = mx.matmul(a, b, stream=d1)
  for _ in range(500):
      b = mx.exp(b, stream=d2)
  return x, b

a = mx.random.uniform(shape=(4096, 512))
b = mx.random.uniform(shape=(512, 4))

import numpy
import mlx.core as mx
import time
import matplotlib.pyplot as plt
from mlx.core.fft import fft

array_sizes = [2**10, 2**12, 2**14, 2**16, 2**18, 2**20]
iterations = 100

cpu_times = []
gpu_times = []

for size in array_sizes:
    print(f"Array Size: {size}")

    real_part = mx.random.normal(shape=(size,), dtype=mx.float32)
    imag_part = mx.random.normal(shape=(size,), dtype=mx.float32)
    
    x = real_part + 1j * imag_part #combines them into a complex64 array
    print(x.dtype)
    gpu_time = 0
    start_time = time.perf_counter()
    for _ in range(iterations):
        d = fft(x, stream=mx.gpu)
    #   start_time = time.perf_counter()
        mx.eval(d) #fft on GPU
    #    gpu_time += time.perf_counter() - start_time
    gpu_time = time.perf_counter() - start_time
    gpu_times.append(gpu_time / iterations)  #average time for GPU
    
    
    cpu_time = 0
    for _ in range(iterations):
        start_time = time.perf_counter()
        mx.eval(fft(x, stream=mx.cpu))  #fft on CPU
        cpu_time += time.perf_counter() - start_time
    cpu_times.append(cpu_time / iterations)  # average time for CPU

plt.figure(figsize=(10, 6))
plt.plot(array_sizes, gpu_times, label="GPU", marker='x')
plt.plot(array_sizes, cpu_times, label="CPU", marker='o')
plt.xlabel("Array Size")
plt.ylabel("Time (seconds)")
plt.title("Performance of FFT on CPU vs GPU")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(array_sizes, (numpy.array(cpu_times))/(numpy.array(gpu_times)), label="", marker='o')
plt.xlabel("Array Size")
plt.ylabel("Ratio of CPU time to GPU time")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()


#IFFT VERSION
import mlx.core as mx
import time
import matplotlib.pyplot as plt
from mlx.core.fft import ifft 

array_sizes = [2**10, 2**12, 2**14, 2**16, 2**18, 2**20]
iterations = 100

cpu_times = []
gpu_times = []

for size in array_sizes:
    print(f"Array Size: {size}")
    
    real_part = mx.random.normal(shape=(size,), dtype=mx.float32)
    imag_part = mx.random.normal(shape=(size,), dtype=mx.float32)
    
    x = real_part + 1j * imag_part

    gpu_time = 0
    for _ in range(iterations):
        start_time = time.perf_counter()
        q = mx.multiply(mx.conj(x), x, stream=mx.gpu)
        mx.eval(ifft(q, stream=mx.gpu))
        gpu_time += time.perf_counter() - start_time
    gpu_times.append(gpu_time / iterations)

    cpu_time = 0
    for _ in range(iterations):
        start_time = time.perf_counter()
        q = mx.multiply(mx.conj(x), x, stream=mx.cpu)
        mx.eval(ifft(q, stream=mx.cpu))
        cpu_time += time.perf_counter() - start_time
    cpu_times.append(cpu_time / iterations)

plt.figure(figsize=(10, 6))
plt.plot(array_sizes, gpu_times, label="GPU", marker='x')
plt.plot(array_sizes, cpu_times, label="CPU", marker='o')
plt.xlabel("Array Size")
plt.ylabel("Time (seconds)")
plt.title("Performance of IFFT on CPU vs GPU")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(array_sizes, (numpy.array(cpu_times))/(numpy.array(gpu_times)), label="", marker='o')
plt.xlabel("Array Size")
plt.ylabel("Ratio of CPU time to GPU time")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()