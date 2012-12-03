import numpy as np

def percent_equal(a, b):
    num_equal = sum(map(lambda x,y: 1 if np.equal(x,y).all() else 0, a, b))
    return float(num_equal) / len(a)

def map_to_max_binary_result(h_x):
    maxes = map(max, h_x)
    res = []
    for i in range(len(maxes)):
        res.append(h_x[i].tolist().index(maxes[i]))
    return all_to_sparse(res, h_x.shape[1])
    
def convert_to_sparse(ex, num_class):
    res = [0] * num_class
    res[ex] = 1
    return res

def all_to_sparse(exs, num_class):
    return map(convert_to_sparse, exs, [num_class] * len(exs))
