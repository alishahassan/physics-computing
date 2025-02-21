import multiprocessing

def multiply (a, b, result_queue):
    result = a * b
    result_queue.put(result)

if __name__ == '__main__':
    result_queue = multiprocessing.Queue()
    num1 = 5
    num2 = 10
    process = multiprocessing.Process(target=multiply, args=(num1, num2, result_queue))
    process.start()
    process.join()
    result = result_queue.get()
    print(f"{num1} * {num2} is: {result}")
