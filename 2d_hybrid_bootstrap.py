import numpy as np
from keras import backend as K
from theano import tensor as T
from keras.engine.topology import Layer
from keras.initializers import Constant

def hybo_2d_conv(x, p, shift, seed = None, unif = True, just_dropout = False):
    '''Theano hybrid bootstrap backend'''
    if p.get_value() < 0. or p.get_value() > 1:
        raise Exception('Hybrid bootstrap p must be in interval [0, 1].')

    if seed is None:
        seed = np.random.randint(1, 10e6)
        rng = K.RandomStreams(seed = seed)

    if(unif == True):
        retain_prob = 1. - rng.uniform((x.shape[0],), 0, p, dtype = x.dtype)
        for dim in range(x.ndim - 1):
            retain_prob = K.expand_dims(retain_prob, dim + 1)
    else:
        retain_prob = 1. - p
        
    mask = rng.binomial((x.shape[0], x.shape[1], x.shape[2], 1), p=retain_prob, dtype=x.dtype)
    mask = T.extra_ops.repeat(mask, x.shape[3], axis = 3)

    if just_dropout:
        x = x * mask / retain_prob
    else:
        x = x * mask + (1 - mask) * T.roll(x, shift = shift, axis = 0)

    return x



class HB_2d_conv(Layer):
    '''Applies the hybrid bootstrap to the input.
        # Arguments
        p: float between 0 and 1. Fraction of the input units to resample if unif = F,
           maximum fraction if unif = T
        shift: int. Should be smaller than batch size.
        unif: bool.  Should p be sampled from unif(0, p)?
        just_dropout: Should we just do dropout?
        '''
    def __init__(self, p, shift, unif = True, just_dropout = False, **kwargs):
        self.init_p = p
        self.shift = shift
        self.unif = unif
        self.just_dropout = just_dropout
        self.uses_learning_phase = True
        self.supports_masking = True
        super(HB_2d_conv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.p = self.add_weight(shape=(),
                                 name = 'p',
                                 initializer=Constant(value = self.init_p),
                                 trainable=False)
        super(HB_2d_conv, self).build(input_shape)

    def call(self, x, mask=None):
        if 0. < self.p.get_value() < 1.:
            x = K.in_train_phase(hybo_2d_conv(x, p = self.p, shift = self.shift,
                                      unif = self.unif,
                                      just_dropout = self.just_dropout), x)
        return x

    def get_config(self):
        config = {'init_p': self.init_p, 'p': self.p, 'shift': self.shift,
                  'unif': self.unif, 'just_dropout': self.just_dropout}
        base_config = super(HB_2d_conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))