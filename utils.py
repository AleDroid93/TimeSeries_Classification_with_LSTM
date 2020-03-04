import numpy as np
from constants import N_TIMESTAMPS

def reshapeToTensor(x):
    new_data = []
    for row in x:
        new_row = np.split(row, N_TIMESTAMPS)
        new_data.append(np.array(new_row))
    return np.array(new_data)

def getBatch(X, i, batch_size):
    start_id = i*batch_size
    end_id = min( (i+1) * batch_size, X.shape[0])
    batch_x = X[start_id:end_id]
    return batch_x
