import theano
import theano.tensor as T
import numpy as np


def score_gaussian(x, mu, A):   # A, inverse of covariance matrix
    return -T.dot(x-mu.reshape((1, -1)), A).astype(theano.config.floatX)

def logp_gaussian(x, mu, A):
    x = x - mu
    logpdf = -0.5 * T.sum(T.dot(x, A) * x, axis=1)
    return logpdf



