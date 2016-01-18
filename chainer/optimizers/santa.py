import math

import numpy

from chainer import cuda
from chainer import optimizer

class Santa(optimizer.GradientMethod):

    """Santa optimization algorithm.

    See: http://arxiv.org/abs/1512.07962v1

    """
    def __init__(self, eta=0.001, sigma=0.999, eps=0.001, C=0.001, gamma=0.5, burnin=2000):
        # note: gamma and burnin are related. When gamma ** t -> 0 then its eqla
        self.eta = eta
        self.sigma = sigma
        self.eps = eps
        self.C = C
        self.gamma = gamma
        self.burnin = burnin

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        state['v'] = xp.zeros_like(param.data)
        state['u'] = numpy.sqrt(self.eta) * numpy.random.normal(size=param.data.shape)
        state['g'] = xp.ones_like(param.data)
        state['alpha'] = xp.full_like(param.data, self.C)

    def force_not_too_small(self, x):
        return numpy.sign(x) * (numpy.abs(x) + 0.01)

    def update_one_cpu(self, param, state):
        v, alpha, prev_u = state['v'], state['alpha'], state['u']
        grad = param.grad
        v *= self.sigma
        v += (1 - self.sigma) * grad * grad

        g = 1 / numpy.sqrt(numpy.sqrt(v) + self.eps)
        param.data += g * prev_u / 2
        if self.t < self.burnin:
            # exploration
            inv_beta = self.gamma ** self.t
            prev_g = state['g']
            alpha += (prev_u * prev_u - self.eta * inv_beta) / 2
            u = numpy.exp(-alpha/2) * prev_u
            u += - g * grad * self.eta + numpy.sqrt(2 * prev_g * self.eta * inv_beta) * numpy.random.normal(size=param.data.shape)
            u += self.eta * inv_beta * (1 - g / self.force_not_too_small(prev_g)) / self.force_not_too_small(prev_u)
            u *= numpy.exp(-alpha/2) 
            alpha += (u * u - self.eta * inv_beta) / 2
            state['g'] = g
            state['u'] = u
        else:
            # refinement
            u = prev_u
            u *= numpy.exp(-alpha/2)
            u -= g * grad * self.eta
            u *= numpy.exp(-alpha/2)
        param.data += g * u /2



    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps',
            'T param, T m, T v',
            '''m += one_minus_beta1 * (grad - m);
               v += one_minus_beta2 * (grad * grad - v);
               param -= lr * m / (sqrt(v) + eps);''',
            'adam')(param.grad, self.lr, 1 - beta1, 1 - beta2,
                    self.eps, param.data, state['m'], state['v'])

