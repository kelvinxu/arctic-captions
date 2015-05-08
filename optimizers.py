import theano
import theano.tensor as tensor

import numpy


def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

"""
General Optimizer Structure: (adadelta, adam, rmsprop, sgd)
Parameters
----------
    lr : theano shared variable
        learning rate, currently only necessaary for sgd
    tparams : OrderedDict()
        dictionary of shared variables {name: variable}
    grads : 
        dictionary of gradients
    inputs :
        inputs required to compute gradients
    cost : 
        objective of optimization
    hard_attn_up :
        additional updates required for hard attention mechanism learning 
Returns
-------
    f_grad_shared : compute cost, update optimizer shared variables
    f_update : update parameters
"""
# See "ADADELTA: An adaptive learning rate method", Matt Zeiler (2012) arXiv
# preprint http://arxiv.org/abs/1212.5701
def adadelta(lr, tparams, grads, inp, cost, hard_attn_up):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up+hard_attn_up, profile=False)
    
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

#    See Lecture 6.5, Coursera: Neural Networks for Machine Learning (2012),
#    Tieleman, T. and Hinton. G. for original methods
#
#    This implementation (with Nesterov Momentum) is described well in:
#    "Generating Sequences with Recurrent Neural Networks", Alex Graves, arxiv preprint
#    http://arxiv.org/abs/1308.0850
def rmsprop(lr, tparams, grads, inp, cost, hard_attn_up):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up+hard_attn_up, profile=False)

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

# See "Adam: A Method for Stochastic Optimization" Kingma et al. (ICLR 2015)
# Theano implementation adapted from Soren Kaae Sonderby (https://github.com/skaae)
# preprint: http://arxiv.org/abs/1412.6980
def adam(lr, tparams, grads, inp, cost, hard_attn_up):
    gshared = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup+hard_attn_up)
    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8
    updates = []
    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * numpy.float32(0.))
        v = theano.shared(p.get_value() * numpy.float32(0.))
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update

# Vanilla SGD
def sgd(lr, tparams, grads, inp, cost, hard_attn_up):
    gshared = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup+hard_attn_up, profile=False)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=False)

    return f_grad_shared, f_update
