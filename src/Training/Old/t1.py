from timeit import default_timer as timer
import os

print('import pytorch')
start = timer()
import torch

end = timer()
print('Elapsed time: ' + str(end - start))

print('import tensorflow')
start = timer()
import tensorflow

end = timer()
print('Elapsed time: ' + str(end - start))