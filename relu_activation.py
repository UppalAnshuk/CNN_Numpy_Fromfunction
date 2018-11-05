import numpy as np

def relu(x):
    return np.maximum(x,0)

"""
a=np.array([1,2,3,-4,0,1])
print(relu(a))
"""