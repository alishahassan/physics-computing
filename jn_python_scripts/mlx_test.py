import mlx.core as mx
import numpy as np

a = mx.array([1, 2, 3, 4])
print(a.shape)
print(a.dtype)

c = mx.array([1,2,3], mx.float32)
d = mx.array([4,5,6], mx.float32)
e = mx.matmul(c,d)
print(e)

#   %%timeit
e = mx.eval(mx.matmul(c,d))

#SCRIPT
# calculate the dot product using numpy 
x = np.ones(10**6)
dot_product = np.dot(x, x)
print(dot_product)
#   %timeit np.dot(x, x)

#SCRIPT
#Calculate the dot product using numpy 
x = mx.ones(10**6)
dot_product = np.dot(x, x)
print(dot_product)
#   %timeit mx.dot(x, x)

x = mx.ones(10**6)
#   %timeit mx.eval(mx.tensordot(x, x, 1))

x.shape

#CPU
x = mx.ones(10**7)
#   %timeit mx.eval(mx.add(x, x, stream=mx.cpu))

#GPU
x = mx.ones(10**7)
#   %timeit mx.eval(mx.add(x, x, stream=mx.gpu))
x.dtype

#TESTING FROM EXAMPLES:
def fun(x):
    return x.flatten(0, 1)

compiled_fun = mx.compile(fun, shapeless=True)

x = mx.random.uniform(shape=(2, 3, 4))

out = compiled_fun(x)

x = mx.random.uniform(shape=(5, 5, 3))

# Ok
out = compiled_fun(x)
