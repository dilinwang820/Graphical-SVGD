import sys
import numpy as np

import theano
import theano.tensor as T

from updates import Adagrad
from theano_utils import floatX, sharedX
from ops import svgd
from rng import np_rng
from evaluate import comm_func_eval

from scipy.stats import multivariate_normal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import score_gaussian, logp_gaussian
import scipy.io as sio

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--grid', type=int, required=False, default=5)
parser.add_argument('--n_samples', type=int, required=False, default=100)
args = parser.parse_args()

import networkx as nx


def init_model():

    G = nx.grid_2d_graph(args.grid, args.grid)
    A = np.asarray(nx.adjacency_matrix(G).todense())

    #A = A * np_rng.uniform(-0.1, 0.1, size=A.shape)
    noise = np.tril( np_rng.uniform(-0.1, 0.1, size=A.shape) )
    noise = noise + noise.T
    A = A * noise

    np.fill_diagonal(A, np.sum(np.abs(A), axis=1) + .1)

    if not np.all(np.linalg.eigvals(A) > 0) or not np.all(np.linalg.eigvals(np.linalg.inv(A)) > 0):
        raise NotImplementedError

    b = np_rng.normal(0, 1, size=(args.grid**2, 1))

    mu0 = sharedX( np.dot(np.linalg.inv(A), b).flatten() )
    A0 = sharedX(A)

    model_params = {'mu':mu0, 'A':A0}

    ## score function
    score_q = score_gaussian

    ## ground truth samples
    gt = np.random.multivariate_normal(mean=mu0.get_value().flatten(), cov=np.linalg.inv(A), size=(10000,), check_valid='raise')

    ## adjacency matrix
    W = np.zeros(A.shape).astype(int)
    W[A != 0] = 1

    assert np.all(np.sum(W, axis=1) > 0), 'illegal inputs'
    assert np.sum((W - W.T)**2) < 1e-8, 'illegal inputs'
    return model_params, score_q, gt,  W

all_algorithms = ['graphical', 'svgd']
max_iter = 5000

model_params, score_q, gt0, N0 = init_model()
x0 = floatX(np.random.uniform(-5, 5, [args.n_samples, gt0.shape[1]]))

for alg in all_algorithms:

    optimizer = Adagrad(lr=5e-3, alpha=0.9)
    xc = svgd(x0, score_q, max_iter=max_iter, alg=alg, N0=N0, optimizer=optimizer, trace=False, **model_params)

    print alg, comm_func_eval(xc, gt0)


