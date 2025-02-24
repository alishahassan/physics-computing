import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import time

array_size = [10**7, 2*10**7, 5*10**7, 10**8]

cpu_times = []
gpu_times = []

for size in array_size: #loops through the array sizes and measure the time for CPU and GPU
    print(f"Array Size: {size}")
    x = mx.ones(size) #creates array 'x' filled with ones of current size
    
    start_time = time.time()
    mx.eval(mx.add(x, x, stream=mx.gpu)) #performs operation (x+x)
    gpu_times.append(time.time() - start_time) #after operation, time taken for GPU is calculated from subtracting start time from current time 

    start_time = time.time()
    mx.eval(mx.add(x, x, stream=mx.cpu))
    cpu_times.append(time.time() - start_time)

plt.figure(figsize=(10, 6))
plt.plot(array_size, gpu_times, label="GPU", marker='x')
plt.plot(array_size, cpu_times, label="CPU", marker='o')
plt.xlabel("Array Size")
plt.ylabel("Time (seconds)")
plt.title("Performance of CPU vs GPU")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()

import mlx.core as mx
import matplotlib.pyplot as plt

array_sizes = [10**6, 10**7, 2 * 10**7, 5 * 10**7]

cpu_times = []
gpu_times = []

for size in array_sizes: #loops through the array sizes and measure the time for CPU and GPU
    print(f"Array Size: {size}")

    x = mx.ones(size) #creates array 'x' filled with ones of current size
    
    print("GPU Time: ")
    gpu_time = %timeit -o mx.eval(mx.add(x, x, stream=mx.gpu)) #measures how long it takes to run the operation mx.add(x, x)
    gpu_times.append(gpu_time.best) #stores the best time from results
    
    print("CPU Time: ")
    cpu_time = %timeit -o mx.eval(mx.add(x, x, stream=mx.cpu))
    cpu_times.append(cpu_time.best)

plt.figure(figsize=(10, 6))
plt.plot(array_sizes, gpu_times, label="GPU", marker='x')
plt.plot(array_sizes, cpu_times, label="CPU", marker='o')
plt.xlabel("Array Size")
plt.ylabel("Time (seconds)")
plt.title("Performance of CPU vs GPU")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import time

array_size = [10**7, 2*10**7, 5*10**7, 10**8]
niterations = 1000
cpu_times = []
gpu_times = []

for size in array_size: #loops through the array sizes and measure the time for CPU and GPU
    print(f"Array Size: {size}")
    x = mx.ones(size, stream=mx.cpu) #creates array 'x' filled with ones of current size
    b = mx.ones(size, stream=mx.cpu)
    
    for i in range(niterations):
    	b = mx.add(b, x, stream=mx.cpu)
        
    start_time = time.time()
    mx.eval(b)
    end_time = time.time()
    dur = (end_time - start_time) / niterations
    cpu_times.append(dur)

    x = mx.ones(size, stream=mx.gpu) #creates array 'x' filled with ones of current size
    b = mx.ones(size, stream=mx.gpu)
    
    for i in range(niterations):
    	b = mx.add(b, x, stream=mx.gpu)
        
    start_time = time.time()
    mx.eval(b)
    end_time = time.time()   
    dur = (end_time - start_time) / niterations
    gpu_times.append(dur)


plt.figure(figsize=(10, 6))
plt.plot(array_size, gpu_times, label="GPU", marker='x')
plt.plot(array_size, cpu_times, label="CPU", marker='o')
plt.xlabel("Array Size")
plt.ylabel("Time (seconds)")
plt.title("Performance of CPU vs GPU")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()

#FFT
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import time

array_sizes = [2**i for i in range(10, 26)] #array sizes from 2^10 to 2^25

cpu_times = []
gpu_times = []

iterations = 5

for size in array_sizes:
    print(f"Array Size: {size}")
    
    x = mx.random.normal(shape=(size,), dtype=mx.float32) #creates complex arrays of size 'size' with dtype 'complex64
    
    gpu_time = 0 #timing GPU
    for _ in range(iterations):
        start_time = time.perf_counter()  #higher precision
        mx.eval(mx.fft(x, stream=mx.gpu))  #FFT operation on GPU
        gpu_time += time.perf_counter() - start_time
    
    gpu_times.append(gpu_time / iterations) #average time for GPU

    cpu_time = 0 #timing CPU
    for _ in range(iterations):
        start_time = time.perf_counter()
        mx.eval(mx.fft(x, stream=mx.cpu))
        cpu_time += time.perf_counter() - start_time
    
    cpu_times.append(cpu_time / iterations) #average time for CPU

plt.figure(figsize=(10, 6))
plt.plot(array_sizes, gpu_times, label="GPU", marker='x')
plt.plot(array_sizes, cpu_times, label="CPU", marker='o')
plt.xlabel("Array Size (log scale)")
plt.ylabel("Time (seconds) (log scale)")
plt.title("Performance of FFT on CPU vs GPU")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()
