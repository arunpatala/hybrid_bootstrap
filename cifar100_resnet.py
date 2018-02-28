import numpy as np
from keras import backend as K
from theano import tensor as T
from keras.engine.topology import Layer
from keras.initializers import Constant

# Theano has much faster batch normalization if
# data format is channels first.

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
    mask = rng.binomial((x.shape[0], 1, x.shape[2], x.shape[3]), p=retain_prob, dtype=x.dtype)
    mask = T.extra_ops.repeat(mask, x.shape[1], axis = 1)
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


import numpy as np
np.random.seed(42)  # for reproducibility
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import add, concatenate
from keras import regularizers
from keras import backend as K
from keras.layers import Lambda
from keras.datasets import cifar100
from keras.callbacks import LearningRateScheduler


def schedule(x):
    x = np.array(x, dtype = 'float32')
    lr = np.piecewise(x,[x <= 60,
                         (x > 60) & (x <= 120),
                         (x > 120) & (x <= 180),
                         (x > 180) & (x <= 240),
                         (x > 240) & (x <= 300),
                         x > 300],
                      [0.1,
                       0.1 * 0.2,
                       0.1 * 0.2 ** 2,
                       0.1 * 0.2 ** 3,
                       0.1 * 0.2 ** 4,
                       0.1 * 0.2 ** 5])
    return(float(lr))

optimizer_schedule = LearningRateScheduler(schedule)
batch_size = 128
nb_epoch = 360
hbp = 0.45
wd = 0.0005
nb_filters = 160
nb_conv = 3

(X_train_s, y_train_s), (X_test, y_test) = cifar100.load_data()
img_rows = X_train_s.shape[1]
img_cols = X_train_s.shape[2]
nb_classes = np.max(y_train_s) + 1

train_mean = np.mean(X_train_s, axis = (0, 1, 2)).reshape((1, 1, 1, 3))
train_sd = np.std(X_train_s, axis = (0, 1, 2)).reshape((1, 1, 1, 3))

X_train_s = (X_train_s - train_mean) / train_sd
X_test = (X_test - train_mean) / train_sd

# Change to channels first format for Theano
X_train_s = np.moveaxis(X_train_s, [1, 2, 3], [2, 3, 1])
X_test = np.moveaxis(X_test, [1, 2, 3], [2, 3, 1])

X_train_s = X_train_s.astype('float32')
X_test = X_test.astype('float32')

# convert class vectors to binary class matrices
Y_train_s = np_utils.to_categorical(y_train_s, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

def rectifier(x):
    return K.relu(x)

def build_resnet_block(inp, nb_reps = 1, nb_conv = 3, nb_filters = 32,
                       hbp = 0.01, wd = 0.0001, stride = 1,
                       data_format = None):
    through = BatchNormalization(axis = 1)(inp)
    through = Lambda(rectifier)(through)
    skip = Conv2D(nb_filters, kernel_size = (1, 1), padding = 'same',
                  strides = (stride, stride), kernel_regularizer = regularizers.l2(wd),
                  use_bias = False, data_format = data_format)(through)
    through = Conv2D(nb_filters, kernel_size = (nb_conv, nb_conv),
                     padding = 'same',
                     strides = (stride, stride),
                     kernel_regularizer = regularizers.l2(wd),
                     use_bias = False, data_format = data_format)(through)
    through = BatchNormalization(axis = 1)(through)
    through = Lambda(rectifier)(through)
    through = Conv2D(nb_filters, kernel_size = (nb_conv, nb_conv),
                     padding = 'same',
                     strides = (1, 1), kernel_regularizer = regularizers.l2(wd),
                     use_bias = False, data_format = data_format)(through)
    inp = add([through, skip])
    for i in range(nb_reps):
        through = BatchNormalization(axis = 1)(inp)
        through = Lambda(rectifier)(through)
        through = Conv2D(nb_filters, kernel_size = (nb_conv, nb_conv), padding = 'same',
                         kernel_regularizer = regularizers.l2(wd),
                         use_bias = False, data_format = data_format)(through)
        through = BatchNormalization(axis = 1)(through)
        through = Lambda(rectifier)(through)
        through = Conv2D(nb_filters, kernel_size = (nb_conv, nb_conv),
                         padding = 'same', kernel_regularizer = regularizers.l2(wd),
                         use_bias = False, data_format = data_format)(through)
        inp = add([through, inp])
    return(inp)

data_format = "channels_first"

inp = Input(shape=(3, img_rows, img_cols))
exp = Conv2D(filters = 160,
             kernel_size = (nb_conv, nb_conv),
             padding = 'same',
             kernel_regularizer = regularizers.l2(wd),
             data_format = data_format,
             use_bias = False)(inp)
hb1 = HB_2d_conv(hbp, shift = -1)(exp)
block1 = build_resnet_block(hb1, nb_reps = 3, nb_conv = nb_conv,
                            nb_filters = nb_filters,
                            hbp = hbp, wd = wd, data_format = data_format)
hb2 = HB_2d_conv(hbp, shift = -2)(block1)
block2 = build_resnet_block(hb2, nb_reps = 3, nb_conv = nb_conv,
                            nb_filters = 2 * nb_filters,
                            hbp = hbp, wd = wd, stride = 2,
                            data_format = data_format)
hb3 = HB_2d_conv(hbp, shift = -3)(block2)
block3 = build_resnet_block(hb3, nb_reps = 3,
                            nb_conv = nb_conv, nb_filters = 4 * nb_filters,
                            hbp = hbp, wd = wd, stride = 2,
                            data_format = data_format)
bn_pre_pool = BatchNormalization(axis = 1)(block3)
rectifier_pre_pool = Lambda(rectifier)(bn_pre_pool)
pool = AveragePooling2D(pool_size=(8, 8), data_format = data_format)(rectifier_pre_pool)
flat = Flatten()(pool)
dense = Dense(nb_classes, activation = 'softmax',
               kernel_regularizer = regularizers.l2(wd),
              use_bias = False)(flat)
model = Model(inputs = inp, outputs = dense)

datagen = ImageDataGenerator(width_shift_range = 0.15,
                             height_shift_range = 0.15,
                             zoom_range = 0.0,
                             rotation_range = 0,
                             horizontal_flip = True,
                             fill_mode='reflect',
                             data_format = data_format)

sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0.00, nesterov = True)
model.compile(loss = 'categorical_crossentropy',
              optimizer = sgd,
              metrics=['accuracy'])

fit = model.fit_generator(datagen.flow(X_train_s, Y_train_s,
                                       batch_size = batch_size),
                          steps_per_epoch = np.ceil(1. * X_train_s.shape[0] / batch_size),
                          epochs = 360,
                          verbose = 1, validation_data = (X_test, Y_test),
                          callbacks = [optimizer_schedule])

return(format(100 * (1 - fit.history['val_acc'][-1]), '.2f'))