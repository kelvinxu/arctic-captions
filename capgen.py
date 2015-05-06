'''
Source code for an attention based image caption generation system described in:

Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
International Conference for Machine Learning (2015)

Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho,
Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, and Yoshua Bengio
University of Toronto/University of Montreal
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy
import copy
import os
import time

from scipy import optimize, stats
from collections import OrderedDict
from sklearn.cross_validation import KFold

import warnings

# see note XXX from (Xu et al. 2015)
from homogeneous_data import HomogeneousData

# dataset iterators
import flickr8k
import flickr30k
import coco


# datasets: 'name', 'load_data: returns iterator', 'prepare_data: some preprocessing'
datasets = {'flickr8k': (flickr8k.load_data, flickr8k.prepare_data),
            'flickr30k': (flickr30k.load_data, flickr30k.prepare_data),
            'coco': (coco.load_data, coco.prepare_data)}

def get_dataset(name):
    return datasets[name][0], datasets[name][1]

'''
Theano requires shared variables, so to
make this code more portable, these two functions
push and pull variables between a shared
variable dictionary and a regular numpy 
dictionary
'''
# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# dropout in theano
def dropout_layer(state_before, use_noise, trng):
    # use_noise is set during training/testing
    # dynamically changing the theano graph
    proj = tensor.switch(use_noise, 
                         state_before * 
                         trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                         state_before * 0.5)
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

# all parameters
def init_params(options):
    params = OrderedDict()
    # embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])
    ctx_dim = options['ctx_dim']
    if options['lstm_encoder']:
        # encoder: LSTM
        params = get_layer('lstm')[0](options, params, prefix='encoder', 
                                      nin=options['ctx_dim'], dim=options['dim'])
        params = get_layer('lstm')[0](options, params, prefix='encoder_rev', 
                                      nin=options['ctx_dim'], dim=options['dim'])
        ctx_dim = options['dim'] * 2
    # init_state, init_cell
    for lidx in xrange(1, options['n_layers_init']):
        params = get_layer('ff')[0](options, params, prefix='ff_init_%d'%lidx, nin=ctx_dim, nout=ctx_dim)
    params = get_layer('ff')[0](options, params, prefix='ff_state', nin=ctx_dim, nout=options['dim'])
    params = get_layer('ff')[0](options, params, prefix='ff_memory', nin=ctx_dim, nout=options['dim'])
    # decoder: LSTM
    params = get_layer('lstm_cond')[0](options, params, prefix='decoder', 
                                       nin=options['dim_word'], dim=options['dim'], 
                                       dimctx=ctx_dim)
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            params = get_layer('ff')[0](options, params, prefix='ff_state_%d'%lidx, nin=options['ctx_dim'], nout=options['dim'])
            params = get_layer('ff')[0](options, params, prefix='ff_memory_%d'%lidx, nin=options['ctx_dim'], nout=options['dim'])
            params = get_layer('lstm_cond')[0](options, params, prefix='decoder_%d'%lidx, 
                                               nin=options['dim'], dim=options['dim'],
                                               dimctx=ctx_dim)
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm', nin=options['dim'], nout=options['dim_word'])
    if options['ctx2out']:
        params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx', nin=ctx_dim, nout=options['dim_word'])
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            params = get_layer('ff')[0](options, params, prefix='ff_logit_h%d'%lidx, nin=options['dim_word'], nout=options['dim_word'])
    params = get_layer('ff')[0](options, params, prefix='ff_logit', nin=options['dim_word'], nout=options['n_words'])

    return params

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive'%kk)
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'), 
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          }

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# some utilities
def ortho_weight(ndim):
    """
    Random orthogonal weights

    Used by norm_weights, in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')

def tanh(x):
    return tensor.tanh(x)

def rectifier(x):
    return tensor.maximum(0., x)

def linear(x):
    return x

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None):
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout, scale=0.01)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# LSTM layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None):
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    # Stack the weight matricies for faster dot prods
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    return params

# This function implements the lstm fprop
def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    dim = tparams[_p(prefix,'U')].shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        init_state = tensor.alloc(0., n_samples, dim)
        init_memory = tensor.alloc(0., n_samples, dim)
    else:
        n_samples = 1
        init_state = tensor.alloc(0., dim)
        init_memory = tensor.alloc(0., dim)

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        elif _x.ndim == 2:
            return _x[:, n*dim:(n+1)*dim]
        return _x[n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tparams[_p(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        h = o * tensor.tanh(c)

        return h, c, i, f, o, preact

    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    rval, updates = theano.scan(_step, 
                                sequences=[mask, state_below],
                                outputs_info = [init_state, init_memory, None, None, None, None],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps, profile=False)
    return rval

# Conditional LSTM layer with Attention
def param_init_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None, dimctx=None):
    if nin == None:
        nin = options['dim']
    if dim == None:
        dim = options['dim']
    if dimctx == None:
        dimctx = options['dim']
    # input to LSTM
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W

    # LSTM to LSTM
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    # bias to LSTM
    params[_p(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx,dim*4)
    params[_p(prefix,'Wc')] = Wc

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx, ortho=False)
    params[_p(prefix,'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[_p(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix,'b_att')] = b_att

    # deeper attention
    if options['n_layers_att'] > 1:
        for lidx in xrange(1, options['n_layers_att']):
            params[_p(prefix,'W_att_%d'%lidx)] = ortho_weight(dimctx)
            params[_p(prefix,'b_att_%d'%lidx)] = numpy.zeros((dimctx,)).astype('float32')

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[_p(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    if options['selector']:
        # attention: selector
        W_sel = norm_weight(dim, 1)
        params[_p(prefix, 'W_sel')] = W_sel
        b_sel = numpy.float32(0.)
        params[_p(prefix, 'b_sel')] = b_sel

    return params

def lstm_cond_layer(tparams, state_below, options, prefix='lstm', 
                    mask=None, context=None, one_step=False, 
                    init_memory=None, init_state=None, 
                    trng=None, use_noise=None, sampling=True,
                    argmax=False, **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_memory, 'previous memory must be provided'
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'U')].shape[0]

    # initial/previous state
    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)
    # initial/previous memory 
    if init_memory == None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # projected context 
    pctx_ = tensor.dot(context, tparams[_p(prefix,'Wc_att')]) + tparams[_p(prefix, 'b_att')]
    if options['n_layers_att'] > 1:
        for lidx in xrange(1, options['n_layers_att']):
            pctx_ = tensor.dot(pctx_, tparams[_p(prefix,'W_att_%d'%lidx)])+tparams[_p(prefix, 'b_att_%d'%lidx)]
            # note to self: this used to be options['n_layers_att'] - 1, so no extra non-linearity if n_layers_att < 3
            if lidx < options['n_layers_att']:
                pctx_ = tanh(pctx_)

    # projected x
    # state_below is timesteps*num samples by d in training
    # this is n * d during sampling 
    state_below = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    # hard attention variables
    if options['attn_type'] == 'stochastic':
        temperature = options.get("temperature", 1)
        semi_sampling_p = options.get("semi_sampling_p", 0.5) 
        temperature_c = theano.shared(numpy.float32(temperature), name='temperature_c')
        h_sampling_mask = trng.binomial((1,), p=semi_sampling_p, n=1, dtype=theano.config.floatX).sum()
    
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_, a_, as_, ct_, pctx_, dp_=None, dp_att_=None):
        # attention
        pstate_ = tensor.dot(h_, tparams[_p(prefix,'Wd_att')])
        pctx_ = pctx_ + pstate_[:,None,:] 
        pctx_list = []
        pctx_list.append(pctx_)
        pctx_ = tanh(pctx_)
        alpha = tensor.dot(pctx_, tparams[_p(prefix,'U_att')])+tparams[_p(prefix, 'c_tt')]
        alpha_pre = alpha
        alpha_shp = alpha.shape

        if options['attn_type'] == 'deterministic':
            alpha = tensor.nnet.softmax(alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
            ctx_ = (context * alpha[:,:,None]).sum(1) # current context
            alpha_sample = alpha # you can return something else reasonable here
        else:
            alpha = tensor.nnet.softmax(temperature_c*alpha.reshape([alpha_shp[0],alpha_shp[1]])) # softmax
            # TODO return alpha_sample
            if sampling:
                alpha_sample = h_sampling_mask * trng.multinomial(pvals=alpha,dtype=theano.config.floatX)\
                              + (1.-h_sampling_mask) * alpha
            else:
                if argmax:
                    alpha_sample = tensor.cast(tensor.eq(tensor.arange(alpha_shp[1])[None,:], 
                                            tensor.argmax(alpha,axis=1,keepdims=True)), theano.config.floatX)
                else:
                    alpha_sample = alpha
            ctx_ = (context * alpha_sample[:,:,None]).sum(1) # current context

        if options['selector']:
            sel_ = tensor.nnet.sigmoid(tensor.dot(h_, tparams[_p(prefix, 'W_sel')])+tparams[_p(prefix,'b_sel')])
            sel_ = sel_.reshape([sel_.shape[0]])
            ctx_ = sel_[:,None] * ctx_

        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tensor.dot(ctx_, tparams[_p(prefix, 'Wc')])

        i = _slice(preact, 0, dim)
        f = _slice(preact, 1, dim)
        o = _slice(preact, 2, dim)
        if options['use_dropout_lstm']:
            i = i * _slice(dp_, 0, dim)
            f = f * _slice(dp_, 1, dim)
            o = o * _slice(dp_, 2, dim)
        i = tensor.nnet.sigmoid(i)
        f = tensor.nnet.sigmoid(f)
        o = tensor.nnet.sigmoid(o)
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * tensor.tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        rval = [h, c, alpha, alpha_sample, ctx_]
        if options['selector']:
            rval += [sel_]
        rval += [pstate_, pctx_, i, f, o, preact, alpha_pre]+pctx_list
        return rval

    if options['use_dropout_lstm']:
        if options['selector']:
            _step0 = lambda m_, x_, dp_, h_, c_, a_, as_, ct_, sel_, pctx_: \
                            _step(m_, x_, h_, c_, a_, as_, ct_, pctx_, dp_)
        else:
            _step0 = lambda m_, x_, dp_, h_, c_, a_, as_, ct_, pctx_: \
                            _step(m_, x_, h_, c_, a_, as_, ct_, pctx_, dp_)
        dp_shape = state_below.shape
        if one_step:
            dp_mask = tensor.switch(use_noise, 
                                    trng.binomial((dp_shape[0], 3*dim), 
                                                  p=0.5, n=1, dtype=state_below.dtype),
                                    tensor.alloc(0.5, dp_shape[0], 3 * dim))
        else:
            dp_mask = tensor.switch(use_noise, 
                                    trng.binomial((dp_shape[0], dp_shape[1], 3*dim), 
                                                  p=0.5, n=1, dtype=state_below.dtype),
                                    tensor.alloc(0.5, dp_shape[0], dp_shape[1], 3*dim))
    else:
        if options['selector']:
            _step0 = lambda m_, x_, h_, c_, a_, as_, ct_, sel_, pctx_: _step(m_, x_, h_, c_, a_, as_, ct_, pctx_)
        else:
            _step0 = lambda m_, x_, h_, c_, a_, as_, ct_, pctx_: _step(m_, x_, h_, c_, a_, as_, ct_, pctx_)

    if one_step:
        if options['use_dropout_lstm']:
            if options['selector']:
                rval = _step0(mask, state_below, dp_mask, init_state, init_memory, None, None, None, None, pctx_)
            else:
                rval = _step0(mask, state_below, dp_mask, init_state, init_memory, None, None, None, pctx_)
        else:
            if options['selector']:
                rval = _step0(mask, state_below, init_state, init_memory, None, None, None, None, pctx_)
            else:
                rval = _step0(mask, state_below, init_state, init_memory, None, None, None, pctx_)
        return rval
    else:
        seqs = [mask, state_below]
        if options['use_dropout_lstm']:
            seqs += [dp_mask]
        outputs_info = [init_state, 
                        init_memory, 
                        tensor.alloc(0., n_samples, pctx_.shape[1]),
                        tensor.alloc(0., n_samples, pctx_.shape[1]),
                        tensor.alloc(0., n_samples, context.shape[2])]
        if options['selector']:
            outputs_info += [tensor.alloc(0., n_samples)]
        outputs_info += [None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None] + [None]#*options['n_layers_att']
        rval, updates = theano.scan(_step0, 
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=[pctx_],
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps, profile=False)
        return rval, updates


# build a training model
def build_model(tparams, options, sampling=True):
    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')
    # context: #samples x #annotations x dim
    ctx = tensor.tensor3('ctx', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # index into the word embedding matrix, shift it forward in time
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted
    if options['lstm_encoder']:
        # encoder
        ctx_fwd = get_layer('lstm')[1](tparams, ctx.dimshuffle(1,0,2), 
                                       options, prefix='encoder')[0].dimshuffle(1,0,2)
        ctx_rev = get_layer('lstm')[1](tparams, ctx.dimshuffle(1,0,2)[:,::-1,:], 
                                       options, prefix='encoder_rev')[0][:,::-1,:].dimshuffle(1,0,2)
        ctx0 = tensor.concatenate((ctx_fwd, ctx_rev), axis=2)
    else:
        ctx0 = ctx

    # initial state/cell
    ctx_mean = ctx0.mean(1)
    for lidx in xrange(1, options['n_layers_init']):
        ctx_mean = get_layer('ff')[1](tparams, ctx_mean, options, 
                                      prefix='ff_init_%d'%lidx, activ='rectifier')
        if options['use_dropout']:
            ctx_mean = dropout_layer(ctx_mean, use_noise, trng)

    init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')
    init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')
    # decoder
    attn_updates = []
    proj, updates = get_layer('lstm_cond')[1](tparams, emb, options, 
                                     prefix='decoder', 
                                     mask=mask, context=ctx0, 
                                     one_step=False, 
                                     init_state=init_state,
                                     init_memory=init_memory,
                                     trng=trng,
                                     use_noise=use_noise,
                                     sampling=sampling)
    attn_updates += updates
    proj_h = proj[0]
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state_%d'%lidx, activ='tanh')
            init_memory = get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory_%d'%lidx, activ='tanh')
            proj, updates = get_layer('lstm_cond')[1](tparams, proj_h, options,
                                             prefix='decoder_%d'%lidx,
                                             mask=mask, context=ctx0,
                                             one_step=False,
                                             init_state=init_state,
                                             init_memory=init_memory,
                                             trng=trng,
                                             use_noise=use_noise,
                                             sampling=sampling)
            attn_updates += updates
            proj_h = proj[0]

    alphas = proj[2]
    alpha_sample = proj[3]
    ctxs = proj[4]

    if options['selector']:
        sels = proj[5]

    if options['use_dropout']:
        proj_h = dropout_layer(proj_h, use_noise, trng)

    # compute word probabilities
    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    if options['prev2out']:
        logit += emb 
    if options['ctx2out']:
        logit += get_layer('ff')[1](tparams, ctxs, options, prefix='ff_logit_ctx', activ='linear')
    logit = tanh(logit)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
            if options['use_dropout']:
                logit = dropout_layer(logit, use_noise, trng)

    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))
    # cost
    x_flat = x.flatten()
    p_flat = probs.flatten()
    cost = -tensor.log(p_flat[tensor.arange(x_flat.shape[0])*probs.shape[1]+x_flat]+1e-8)
    cost = cost.reshape([x.shape[0], x.shape[1]])
    masked_cost = cost * mask 
    cost = (masked_cost).sum(0)

    opt_outs = dict() # optional outputs
    if options['selector']:
        opt_outs['selector'] = sels
    if options['attn_type'] == 'stochastic':
        opt_outs['masked_cost'] = masked_cost # need this for reinforce later
        opt_outs['attn_updates'] = attn_updates

    return trng, use_noise, [x, mask, ctx], alphas, alpha_sample, cost, opt_outs

# build a sampler
def build_sampler(tparams, options, use_noise, trng, sampling=True):
    # context: #annotations x dim
    ctx = tensor.matrix('ctx_sampler', dtype='float32')
    if options['lstm_encoder']:
        # encoder
        ctx_fwd = get_layer('lstm')[1](tparams, ctx, 
                                       options, prefix='encoder')[0]
        ctx_rev = get_layer('lstm')[1](tparams, ctx[::-1,:], 
                                       options, prefix='encoder_rev')[0][::-1,:]
        ctx = tensor.concatenate((ctx_fwd, ctx_rev), axis=1)

    # initial state/cell
    ctx_mean = ctx.mean(0)
    for lidx in xrange(1, options['n_layers_init']):
        ctx_mean = get_layer('ff')[1](tparams, ctx_mean, options, 
                                      prefix='ff_init_%d'%lidx, activ='rectifier')
        if options['use_dropout']:
            ctx_mean = dropout_layer(ctx_mean, use_noise, trng)
    init_state = [get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state', activ='tanh')]
    init_memory = [get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory', activ='tanh')]
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state.append(get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_state_%d'%lidx, activ='tanh'))
            init_memory.append(get_layer('ff')[1](tparams, ctx_mean, options, prefix='ff_memory_%d'%lidx, activ='tanh'))

    print 'Building f_init...',
    f_init = theano.function([ctx], [ctx]+init_state+init_memory, name='f_init', profile=False)
    print 'Done'

    # x: 1 x 1
    ctx = tensor.matrix('ctx_sampler', dtype='float32')
    x = tensor.vector('x_sampler', dtype='int64')
    init_state = [tensor.matrix('init_state', dtype='float32')]
    init_memory = [tensor.matrix('init_memory', dtype='float32')]
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            init_state.append(tensor.matrix('init_state', dtype='float32'))
            init_memory.append(tensor.matrix('init_memory', dtype='float32'))

    # if it's the first word, emb should be all zero
    emb = tensor.switch(x[:,None] < 0, tensor.alloc(0., 1, tparams['Wemb'].shape[1]), 
                        tparams['Wemb'][x])

    proj = get_layer('lstm_cond')[1](tparams, emb, options, 
                                     prefix='decoder', 
                                     mask=None, context=ctx, 
                                     one_step=True, 
                                     init_state=init_state[0],
                                     init_memory=init_memory[0],
                                     trng=trng,
                                     use_noise=use_noise,
                                     sampling=sampling)

    next_state, next_memory, ctxs = [proj[0]], [proj[1]], [proj[4]]
    proj_h = proj[0]
    if options['n_layers_lstm'] > 1:
        for lidx in xrange(1, options['n_layers_lstm']):
            proj = get_layer('lstm_cond')[1](tparams, proj_h, options,
                                             prefix='decoder_%d'%lidx,
                                             context=ctx,
                                             one_step=True,
                                             init_state=init_state[lidx],
                                             init_memory=init_memory[lidx],
                                             trng=trng,
                                             use_noise=use_noise,
                                             sampling=sampling)
            next_state.append(proj[0])
            next_memory.append(proj[1])
            ctxs.append(proj[4])
            proj_h = proj[0]

    if options['use_dropout']:
        proj_h = dropout_layer(proj[0], use_noise, trng)
    else:
        proj_h = proj[0]
    logit = get_layer('ff')[1](tparams, proj_h, options, prefix='ff_logit_lstm', activ='linear')
    if options['prev2out']:
        logit += emb 
    if options['ctx2out']:
        logit += get_layer('ff')[1](tparams, ctxs[-1], options, prefix='ff_logit_ctx', activ='linear')
    logit = tanh(logit)
    if options['use_dropout']:
        logit = dropout_layer(logit, use_noise, trng)
    if options['n_layers_out'] > 1:
        for lidx in xrange(1, options['n_layers_out']):
            logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit_h%d'%lidx, activ='rectifier')
            if options['use_dropout']:
                logit = dropout_layer(logit, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options, prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    next_probs = tensor.nnet.softmax(logit)
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # next word probability
    f_next = theano.function([x, ctx]+init_state+init_memory, [next_probs, next_sample]+next_state+next_memory, name='f_next', profile=False)

    return f_init, f_next

# generate sample
def gen_sample(tparams, f_init, f_next, ctx0, options, 
               trng=None, k=1, maxlen=30, stochastic=False):
    if k > 1:
        assert not stochastic, 'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    hyp_memories = []

    rval = f_init(ctx0)
    ctx0 = rval[0]
    next_state = []
    next_memory = []
    for lidx in xrange(options['n_layers_lstm']):
        next_state.append(rval[1+lidx])
        next_state[-1] = next_state[-1].reshape([live_k, next_state[-1].shape[0]])
    for lidx in xrange(options['n_layers_lstm']):
        next_memory.append(rval[1+options['n_layers_lstm']+lidx])
        next_memory[-1] = next_memory[-1].reshape([live_k, next_memory[-1].shape[0]])
    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        rval = f_next(*([next_w, ctx0]+next_state+next_memory))
        next_p = rval[0]
        next_w = rval[1]
        next_state = []
        for lidx in xrange(options['n_layers_lstm']):
            next_state.append(rval[2+lidx])
        next_memory = []
        for lidx in xrange(options['n_layers_lstm']):
            next_memory.append(rval[2+options['n_layers_lstm']+lidx])
        if stochastic:
            sample.append(next_w[0])
            sample_score += next_p[0,next_w[0]]
            if next_w[0] == 0:
                break
        else:
            cand_scores = hyp_scores[:,None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]
            
            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_states.append([])
            new_hyp_memories = []
            for lidx in xrange(options['n_layers_lstm']):
                new_hyp_memories.append([])

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[ti])
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_states[lidx].append(copy.copy(next_state[lidx][ti]))
                for lidx in xrange(options['n_layers_lstm']):
                    new_hyp_memories[lidx].append(copy.copy(next_memory[lidx][ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            for lidx in xrange(options['n_layers_lstm']):
                hyp_states.append([])
            hyp_memories = []
            for lidx in xrange(options['n_layers_lstm']):
                hyp_memories.append([])

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    for lidx in xrange(options['n_layers_lstm']):
                        hyp_states[lidx].append(new_hyp_states[lidx][idx])
                    for lidx in xrange(options['n_layers_lstm']):
                        hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = []
            for lidx in xrange(options['n_layers_lstm']):
                next_state.append(numpy.array(hyp_states[lidx]))
            next_memory = []
            for lidx in xrange(options['n_layers_lstm']):
                next_memory.append(numpy.array(hyp_memories[lidx]))

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


def pred_probs(f_log_probs, options, worddict, prepare_data, data, iterator, verbose=False):
    """ Get log probabilities of captions
    Parameters
    ----------
    f_log_probs : theano function
        compute the log probability of a x given the context
    options : dict
        options dictionary
    worddict : dict
        maps words to one-hot encodings
    prepare_data : function
        see corresponding dataset class for details
    data : numpy array 
        output of load_data, see corresponding dataset class
    iterator : KFold 
        indices from scikit-learn KFold
    verbose : boolean
        if True print progress
    Returns 
    -------
    probs : numpy array
        array of log probabilities indexed by example
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 1)).astype('float32')

    n_done = 0

    for _, valid_index in iterator:
        x, mask, ctx = prepare_data([data[0][t] for t in valid_index], 
                                     data[1], 
                                     worddict,
                                     maxlen=None,
                                     n_words=options['n_words'])
        pred_probs = f_log_probs(x,mask,ctx)
        probs[valid_index] = pred_probs[:,None]

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples computed'%(n_done,n_samples)

    return probs

def validate_options(options):
    # Put friendly reminders here
    if options['dim_word'] > options['dim']:
        warnings.warn('dim_word should only be as large as dim.')

    if options['lstm_encoder']:
        warnings.warn('Note that this is a 1-D bidirectional LSTM, not 2-D one.')

    if options['use_dropout_lstm']:
        warnings.warn('dropout in the lstm seems not to help')

    return options


"""Note: all the hyperparameters are stored in a dictionary model_options (or options outside train).
   train() then proceeds to do the following: 
       1. The params are initialized (or reloaded)
       2. The computations graph is built symbolically using Theano.
       3. A cost is defined, then gradient are obtained through automatically with tensor.grad :D
       4. With some helper functions, gradient descent + periodic saving/printing proceeds 
"""
def train(dim_word=100, # word vector dimensionality
          ctx_dim=512, # context vector dimensionality
          dim=1000, # the number of LSTM units
          attn_type='stochastic', # see section XX from paper
          n_layers_att=1, # number of layers used to compute the attention weights
          n_layers_out=1, # number of layers used to compute logit
          n_layers_lstm=1, # number of lstm layers
          n_layers_init=1, # number of layers to initialize LSTM at time 0
          lstm_encoder=False, # if True, run bidirectional LSTM on input units
          prev2out=False, # Feed previous word into logit
          ctx2out=False, # Feed attention weighted ctx into logit
          alpha_entropy_c=0., # hard attn param
          RL_sumCost=False, # hard attn param
          semi_sampling_p=0.5, # hard attn param
          temperature=1., # hard attn param
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0., # weight decay coeff
          alpha_c=0., # doubly stochastic coeff
          lrate=0.01,  # used only for SGD
          selector=False, # selector (see paper) 
          n_words=10000, # vocab size
          maxlen=100, # maximum length of the description
          optimizer='rmsprop',
          batch_size = 16,
          valid_batch_size = 16,
          saveto='model.npz', # relative path of saved model file
          validFreq=1000,
          saveFreq=1000, # save the parameters after every saveFreq updates
          sampleFreq=100, # generate some samples after every sampleFreq updates
          dataset='flickr8k',
          dictionary=None, # word dictionary
          use_dropout=False, # setting this true turns on dropout at various points
          use_dropout_lstm=False, # dropout on lstm gates
          reload_=False):

    # Model options
    model_options = locals().copy()
    model_options = validate_options(model_options)

    # reload options
    if reload_ and os.path.exists(saveto):
        print "Reloading options"
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'
    load_data, prepare_data = get_dataset(dataset)
    train, valid, test, worddict = load_data()

    # index 0 and 1 always code for the end of sentence and unknown token
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # Initialize (or reload) the parameters using 'model_options'
    # then build the Theano graph
    print 'Building model'
    params = init_params(model_options)
    if reload_ and os.path.exists(saveto):
        print "Reloading model"
        params = load_params(saveto, params)

    # numpy arrays -> theano shared variables
    tparams = init_tparams(params)

    # In order, we get:
    #   1) trng - theano random number generator
    #   2) use_noise - flag that turns on dropout  
    #   3) inps - inputs for f_grad_shared
    #   4) cost - TODO adjust this 
    #   5) opts_out - optional outputs (e.g selector)
    trng, use_noise, \
          inps, alphas, alphas_sample,\
          cost, \
          opt_outs = \
          build_model(tparams, model_options) 


    # To sample, we use beam search: 1) f_init is a function that initializes
    # the LSTM at time 0 (see eq. XXX), 2) f_next returns the distribution over
    # words and also the new "initial state/memory" see equation 
    print 'Buliding sampler'
    f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)

    # we want the cost without any the regularizers
    f_log_probs = theano.function(inps, -cost, profile=False, 
                                        updates=opt_outs['attn_updates']
                                        if model_options['attn_type']=='stochastic'
                                        else None)
  
    cost = cost.mean()
    # add L2 regularization costs
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # Doubly stochastic regularization (see section )
    if alpha_c > 0.:
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * ((1.-alphas.sum(0))**2).sum(0).mean()
        cost += alpha_reg

    hard_attn_updates = []
    # Backprop!
    if model_options['attn_type'] == 'deterministic':
        grads = tensor.grad(cost, wrt=itemlist(tparams))
    else:
        baseline_time = theano.shared(numpy.float32(0.), name='baseline_time') 
        opt_outs['baseline_time'] = baseline_time

        alpha_entropy_c = theano.shared(numpy.float32(alpha_entropy_c), name='alpha_entropy_c')
        alpha_entropy_reg = alpha_entropy_c * (alphas*tensor.log(alphas)).mean()
        if model_options['RL_sumCost']:
            grads = tensor.grad(cost, wrt=itemlist(tparams), 
                                disconnected_inputs='raise', 
                                known_grads={alphas:(baseline_time-opt_outs['masked_cost'].mean(0))[None,:,None]/10.*
                                            (alphas_sample/alphas) + alpha_entropy_c*(tensor.log(alphas) + 1)})
        else:
            grads = tensor.grad(cost, wrt=itemlist(tparams), 
                            disconnected_inputs='raise', 
                            known_grads={alphas:opt_outs['masked_cost'][:,:,None]/10.*
                            (alphas_sample/alphas) + alpha_entropy_c*(tensor.log(alphas) + 1)})
        # create update rules 
        hard_attn_updates += [(baseline_time, baseline_time * 0.9 + 0.1 * opt_outs['masked_cost'].mean())]
        hard_attn_updates += opt_outs['attn_updates']

    # to getthe cost after regularization or the gradients, use this
    # f_cost = theano.function([x, mask, ctx], cost, profile=False)
    # f_grad = theano.function([x, mask, ctx], grads, profile=False)

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost, hard_attn_updates)

    print 'Optimization'

    # See note in section XXX of paper
    train_iter = HomogeneousData(train, batch_size=batch_size, maxlen=maxlen)

    if valid:
        kf_valid = KFold(len(valid[0]), n_folds=len(valid[0])/valid_batch_size, shuffle=False)
    if test:
        kf_test = KFold(len(test[0]), n_folds=len(test[0])/valid_batch_size, shuffle=False)

    # bare-bones training logs
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = numpy.load(saveto)['history_errs'].tolist()
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    uidx = 0
    estop = False
    for eidx in xrange(max_epochs):
        n_samples = 0

        print 'Epoch ', eidx

        for caps in train_iter:
            n_samples += len(caps)
            uidx += 1
            # turn on dropout
            use_noise.set_value(1.)
            
            # preprocess the caption, recording the 
            # time spent to help detect bottlenecks
            pd_start = time.time()
            x, mask, ctx = prepare_data(caps, 
                                        train[1],
                                        worddict,
                                        maxlen=maxlen,
                                        n_words=n_words)
            pd_duration = time.time() - pd_start

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen
                continue
            
            # get the cost for the minibatch, and update the weights
            ud_start = time.time()
            cost = f_grad_shared(x, mask, ctx)
            f_update(lrate)
            ud_duration = time.time() - ud_start

            # Numerical stability check
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'PD ', pd_duration, 'UD ', ud_duration

            # Checkpoint
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if best_p != None:
                    params = copy.copy(best_p)
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                print 'Done'

            # Print a generated sample as a sanity check
            if numpy.mod(uidx, sampleFreq) == 0:
                # turn off dropout first
                use_noise.set_value(0.)
                x_s = x
                mask_s = mask
                ctx_s = ctx
                # generate and decode the a subset of the current training batch
                for jj in xrange(numpy.minimum(10, len(caps))):
                    sample, score = gen_sample(tparams, f_init, f_next, ctx_s[jj], model_options,
                                               trng=trng, k=5, maxlen=30, stochastic=False)
                    # Decode the sample for ease of monitoring
                    print 'Truth ',jj,': ',
                    for vv in x_s[:,jj]:
                        if vv == 0:
                            break
                        if vv in word_idict:
                            print word_idict[vv], 
                        else:
                            print 'UNK',
                    print
                    for kk, ss in enumerate([sample[0]]):
                        print 'Sample (', kk,') ', jj, ': ',
                        for vv in ss:
                            if vv == 0:
                                break
                            if vv in word_idict:
                                print word_idict[vv], 
                            else:
                                print 'UNK',
                    print

            # Log validation loss + checkpoint the model with the best validation log likelihood
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                train_err = 0
                valid_err = 0
                test_err = 0

                if valid:
                    valid_err = -pred_probs(f_log_probs, model_options, worddict, prepare_data, valid, kf_valid).mean()
                if test:
                    test_err = -pred_probs(f_log_probs, model_options, worddict, prepare_data, test, kf_test).mean()

                history_errs.append([valid_err, test_err])
                
                # the model with the best validation long likelihood is saved seperately with a different name
                if uidx == 0 or valid_err <= numpy.array(history_errs)[:,0].min():
                    best_p = unzip(tparams)
                    print 'Saving model with best validation ll'
                    params = copy.copy(best_p)
                    params = unzip(tparams)
                    numpy.savez(saveto+'_bestll', history_errs=history_errs, **params) 
                    bad_counter = 0

                # abort training if perplexity has been increasing for too long
                if eidx > patience and len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:-patience,0].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

        print 'Seen %d samples'%n_samples

        if estop:
            break
            
    # use the best nll parameters for final check point (if they exist)
    if best_p is not None: 
        zipp(best_p, tparams) 

    use_noise.set_value(0.)
    train_err = 0
    valid_err = 0
    test_err = 0
    if valid:
        valid_err = -pred_probs(f_log_probs, model_options, worddict, prepare_data, valid, kf_valid)
    if test:
        test_err = -pred_probs(f_log_probs, model_options, worddict, prepare_data, test, kf_test)


    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p, train_err=train_err, 
                valid_err=valid_err, test_err=test_err, history_errs=history_errs, 
                **params)

    return train_err, valid_err, test_err


if __name__ == '__main__':
    pass
