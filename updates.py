import theano
import theano.tensor as T
import numpy as np
from theano_utils import floatX
from rng import t_rng 


class SGD():

    def __init__(self, lr=0.01, alpha=0.9, decay=True):
        self.__dict__.update(locals())

    def __call__(self, params, grads):
        updates = []

        t_prev = theano.shared(floatX(0.))

        t = t_prev + 1
        for p,g in zip(params, grads):
            value = p.get_value(borrow=True)
            velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
            if self.decay:
                curr_lr = self.lr * T.cast((1 + t)**(-.55), theano.config.floatX)
            else:
                curr_lr = self.lr
            step = self.alpha * velocity + curr_lr * g

            updated_p = p - step 
            updates.append((p, updated_p))
            updates.append((t_prev, t))

        return updates


class Adam():

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, e=1e-8, l=1-1e-8):
        self.__dict__.update(locals())  

    def __call__(self, params, grads):
        updates = []

        t_prev = theano.shared(floatX(0.))

        # Using theano constant to prevent upcasting of float32
        one = T.constant(1)

        t = t_prev + 1
        a_t = self.lr*T.sqrt(one-self.b2**t)/(one-self.b1**t)

        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=param.broadcastable)
            m_t = self.b1*m_prev + (one-self.b1)*g_t
            v_t = self.b2*v_prev + (one-self.b2)*g_t**2
            step = a_t*m_t/(T.sqrt(v_t) + self.e)

            updates.append((m_prev, m_t))
            updates.append((v_prev, v_t))
            updates.append((param, param-step))

        updates.append((t_prev, t))
        return updates



class Adagrad():

    def __init__(self, lr=0.01, alpha=-1, epsilon=1e-6):
        self.__dict__.update(locals())

    def __call__(self, params, grads):
        updates = []

        for p,g in zip(params,grads):
            acc = theano.shared(p.get_value() * 0.)
            if self.alpha > 0:
                acc_t = self.alpha * acc + (1. - self.alpha) * (g ** 2)
            else:
                acc_t = acc + g ** 2
            updates.append((acc, acc_t))

            p_t = p - (self.lr / T.sqrt(acc_t + self.epsilon)) * g
            updates.append((p, p_t))
        return updates  



