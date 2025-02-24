import multiprocessing
from functools import partial

# function that takes a tuple of values
def multiply(pair, multiplier=1):
    # multiplies two numbers in the tuple and applies an optional multiplier.
    a, b = pair  # unpack tuple
    return a * b * multiplier

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    data = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

    multiply_with_multiplier = partial(multiply, multiplier=2)

    result = pool.map(multiply_with_multiplier, data)

    print(result)
