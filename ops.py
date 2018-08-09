import sys
import numpy as np

import theano
import theano.tensor as T
from updates import Adagrad
from tqdm import tqdm
from rng import t_rng, np_rng
from theano_utils import floatX, sharedX


def sqr_dist(x, y, e=1e-8):
    if x.ndim == 2:
        xx = T.sqr(T.sqrt((x*x).sum(axis=1) + e))
        yy = T.sqr(T.sqrt((y*y).sum(axis=1) + e))
        dist = T.dot(x, y.T)
        dist *= -2.
        dist += xx.dimshuffle(0, 'x')
        dist += yy.dimshuffle('x', 0)
    else:
        raise NotImplementedError
    return dist


def median_distance(H, e=1e-6):
    if H.ndim != 2:
        raise NotImplementedError

    V = H.flatten()
    # median distance
    h = T.switch(T.eq((V.shape[0] % 2), 0),
        # if even vector
        T.mean(T.sort(V)[ ((V.shape[0] // 2) - 1) : ((V.shape[0] // 2) + 1) ]),
        # if odd vector
        T.sort(V)[V.shape[0] // 2])
    #h = h / T.log(H.shape[0] + 1).astype(theano.config.floatX)
    return h


def poly_kernel(x, e=1e-8):
    x = x - T.mean(x, axis=0)
    kxy = 1 + T.dot(x, x.T)
    dxkxy = x * x.shape[0].astype(theano.config.floatX)

    return kxy, dxkxy


def rbf_kernel(x):
    H = sqr_dist(x, x)
    h = median_distance(H)

    kxy = T.exp(-H / h)

    dxkxy = -T.dot(kxy, x)
    sumkxy = T.sum(kxy, axis=1).dimshuffle(0, 'x')
    dxkxy = (dxkxy + x * sumkxy) * 2. / h

    return kxy, dxkxy


def graphical_rbf_kernel(x, N):
        
    def graphical_rbf(nbrs_i, i, x):  # i_th dimension, neighbors of x_i
        #xn = x * nbrs_i.reshape((1, -1)).astype(theano.config.floatX)
        selected = T.neq(nbrs_i, 0).nonzero()[0]
        new_i = T.eq(selected, i).nonzero()[0]

        xn = x[:, selected]
        kxy, dxkxy = rbf_kernel(xn)
        #kxy, dxkxy = poly_kernel(xn)
        return kxy, dxkxy[:, new_i].flatten()

    kernel_info, _ = theano.scan(fn=graphical_rbf, outputs_info=None, sequences=[N, T.arange(N.shape[0])], non_sequences=[x])
    kxy, dxkxy = kernel_info
    return kxy, dxkxy 


def graphical_svgd_gradient(x, score_q, N, kernel='rbf', **model_params):

    grad = score_q(x, **model_params)
    if kernel == 'rbf':
        g_kxy, g_dxkxy = graphical_rbf_kernel(x, N)
        grad = grad.dimshuffle(1, 0, 'x')
        #svgd_grad = (T.batched_dot(g_kxy, grad).reshape(g_dxkxy.shape) + g_dxkxy) / T.sum(g_kxy, axis=2)
        svgd_grad = (T.batched_dot(g_kxy, grad).reshape(g_dxkxy.shape) + g_dxkxy) / x.shape[0].astype(theano.config.floatX)
    else:
        raise NotImplementedError

    return svgd_grad.dimshuffle(1, 0)


def svgd_gradient(x, score_q, kernel='rbf', **model_params):

    grad = score_q(x, **model_params)

    if kernel == 'rbf':
        kxy, dxkxy = rbf_kernel(x)
    else:
        raise NotImplementedError

    svgd_grad = (T.dot(kxy, grad) + dxkxy) / T.sum(kxy, axis=1).dimshuffle(0, 'x')
    return svgd_grad




'''
    # x0: initlizations
    # score_q: log probability
    # alg: 'svgd' or 'graphical'
    # N0: adjacency matrix
    # optimizer
'''

def svgd(x0, score_q, max_iter=2000, alg='svgd', N0=None, optimizer=None, progressbar=True, trace=False, **model_params):
    if alg == 'graphical' and N0 is None:
        raise NotImplementedError

    theta = theano.shared(floatX(np.copy(x0).reshape((len(x0), -1)))) # initlization

    if alg == 'graphical':
        N = theano.shared(N0.astype('int32')) # adjacency matrix
        svgd_grad = -1 * graphical_svgd_gradient(theta, score_q, N, **model_params)

    elif alg == 'svgd':
        svgd_grad = -1 * svgd_gradient(theta, score_q, **model_params)

    else:
        raise NotImplementedError

    # Initialize optimizer
    if optimizer is None:
        optimizer = Adagrad(lr=1e-2, alpha=.5)  

    svgd_updates = optimizer([theta], [svgd_grad])

    svgd_step = theano.function([], [], updates=svgd_updates)

    # Run svgd optimization
    if progressbar:
        progress = tqdm(np.arange(max_iter))
    else:
        progress = np.arange(max_iter)

    xx = []
    for ii in progress:
        svgd_step()

        if trace:
            xx.append(theta.get_value())

    theta_val = theta.get_value().reshape(x0.shape)

    if trace:
        return theta_val, np.asarray(xx)
    else:
        return theta_val

