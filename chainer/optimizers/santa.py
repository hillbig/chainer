import math

import numpy

from chainer import cuda
from chainer import optimizer

class Santa(optimizer.GradientMethod):

    """Santa optimization algorithm.

    See: http://arxiv.org/abs/1512.07962v1

    """
    def __init__(self, eta=0.001, sigma=0.999, eps=0.001, C=0.001, gamma=0.5, burnin=1000):
        self.eta = eta
        self.sigma = sigma
        self.eps = eps
        self.C = C
        self.gamma = gamma
        self.burnin = burnin

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        state['v'] = xp.zeros_like(param.data, dtype=xp.float32)
        state['u'] = numpy.sqrt(self.eta) * xp.random.normal(size=param.data.shape, dtype=xp.float32)
        state['g'] = xp.ones_like(param.data, dtype=xp.float32)
        state['alpha'] = xp.full_like(param.data, self.C, dtype=xp.float32)

    def force_not_too_small(self, x):
        return numpy.copysign(numpy.maximum(numpy.abs(x), 0.01), x)

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
        xp = cuda.get_array_module(param.data)
        g = xp.empty(param.data.shape, dtype=xp.float32)

        cuda.elementwise(
            'T grad, T sigma, T eps, T prev_u',
            'T data, T v, T g',
            '''v *= sigma;
               v += (1 - sigma) * grad * grad;
               g = 1 / sqrt(sqrt(v) + eps);
               data += g * prev_u / 2;''',
            'santa_pre')(param.grad, self.sigma, self.eps, state['u'], param.data, state['v'], g)
        if self.t < self.burnin:
            # exploration
            zeta = xp.random.normal(size=param.data.shape, dtype=xp.float32)
            inv_beta = self.gamma ** self.t
            u = xp.empty(param.data.shape, dtype=xp.float32)
            cuda.elementwise(
                'T prev_g, T prev_u, T inv_beta, T eta, T g, T zeta, T grad',
                'T alpha, T u',
                '''alpha += (prev_u * prev_u - eta * inv_beta) / 2;
                   u = exp(-alpha/2) * prev_u;
                   u += -g * grad * eta + sqrt(2 * prev_g * eta * inv_beta) * zeta;
                   T prev_g_fixed = copysign(max(abs(prev_g), 0.01), prev_g);
                   T prev_u_fixed = copysign(max(abs(prev_u), 0.01), prev_u);
                   u += eta * inv_beta * (1 - g / prev_g_fixed) / prev_u_fixed;
                   u *= exp(-alpha/2);
                   alpha += (u * u - eta * inv_beta)/2;
                ''',
                'santa_exploration')(
                    state['g'], state['u'], inv_beta, self.eta, g, zeta, param.grad, state['alpha'], u)
            state['g'] = g
            state['u'] = u
        else:
            # refinement
            cuda.elementwise(
                'T alpha, T g, T grad, T eta',
                'T u',
                '''u *= exp(-alpha/2);
                   u -= g * grad * eta;
                   u *= exp(-alpha/2);''',
                'santa_refinement')(
                    state['alpha'], g, param.grad, self.eta, state['u'])
        param.data += g * state['u'] / 2
                
                

        


