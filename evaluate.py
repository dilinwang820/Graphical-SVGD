import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import theano
import theano.tensor as T
from theano_utils import floatX, sharedX


def comm_func_eval(samples, ground_truth):

    samples = np.copy(samples)
    ground_truth = np.copy(ground_truth)

    def ex():
        f0 = np.mean(samples, axis=0)
        f1 = np.mean(ground_truth, axis=0)
        return np.mean((f0-f1)**2)

    def exsqr():
        f0 = np.mean(samples**2, axis=0)
        f1 = np.mean(ground_truth**2, axis=0)
        return np.mean((f0-f1)**2)


    out = {}
    out['ex'] = ex()
    out['exsqr'] = exsqr()
    return out



