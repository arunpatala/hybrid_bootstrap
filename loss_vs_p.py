import numpy as np
np.random.seed(42)
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.metrics import log_loss
from keras.callbacks import Callback, CSVLogger
from basic_hb import *


class UncorruptedTrainHistory(Callback):
    '''Callback to keep track of uncorrupted training losses'''
    def __init__(self, data):
        self.data_X, self.data_Y = data
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []
    
    def on_epoch_end(self, batch, logs={}):
        loss, acc = self.model.evaluate(self.data_X, self.data_Y)
        print("\nUncorrupted Training Loss: {}, Uncorrupted Training Accuracy: {}" .format(loss, acc))
        self.losses.append(loss)
        self.accs.append(acc)

class OptimizerScheduler(Callback):
    """Schedule for the optimizer.  Virtually identical to keras LearningRateScheduler but
    sets momentum (also I took out the chedule output checker)
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """
    def __init__(self, schedule):
        super(OptimizerScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr, momentum = self.schedule(epoch)
        K.set_value(self.model.optimizer.lr, lr)
        K.set_value(self.model.optimizer.momentum, momentum)
        
D = 1
batch_size = 128
nb_classes = 10
img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv = 5

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

small_indices = np.array([], dtype = 'int32')
for i in range(nb_classes):
    first_100_i_indices = np.arange(X_train.shape[0])[y_train == i][0:100]
    small_indices = np.concatenate([small_indices, first_100_i_indices])
    
small_X_train = X_train[small_indices]
small_y_train = y_train[small_indices]
small_Y_train = Y_train[small_indices]

def schedule(x):
    '''Rather ugly way of defining a learning schedule function'''
    x = np.array(x, dtype = 'float32')
    lr = np.piecewise(x, [x < 50*D, (x >= 50*D) & (x < 100*D), x >= 100*D],
                        [0.01, 0.001, 0.0001])
    momentum = np.piecewise(x, [x < 50*D, x >= 50*D],
                        [0.9, 0.99])
    return((lr, momentum))

frames = []
uncorrupted_train_history = UncorruptedTrainHistory([small_X_train, small_Y_train])
optimizer_scheduler = OptimizerScheduler(schedule)
for just_dropout in [False, True]:
    for unif in [False, True]:
        for hbp in (np.arange(11) / 10.):
            model = Sequential()
            model.add(HB(hbp / 2, shift = -1, input_shape = (img_rows, img_cols, 1), unif = unif, just_dropout = just_dropout))
            model.add(Conv2D(nb_filters, kernel_size = (nb_conv, nb_conv), padding = 'valid', activation = 'relu'))
            model.add(HB(hbp, shift = -2, unif = unif, just_dropout = just_dropout))
            model.add(Conv2D(nb_filters, kernel_size = (nb_conv, nb_conv), padding = 'valid', activation = 'relu'))
            model.add(MaxPooling2D((nb_pool, nb_pool)))
            model.add(HB(hbp, shift = -3, unif = unif, just_dropout = just_dropout))
            model.add(Conv2D(nb_filters, kernel_size = (nb_conv, nb_conv), padding = 'valid', activation = 'relu'))
            model.add(HB(hbp, shift = -4, unif = unif, just_dropout = just_dropout))
            model.add(Conv2D(nb_filters, kernel_size = (nb_conv, nb_conv), padding = 'valid', activation = 'relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Flatten())
            model.add(HB(hbp, shift = -6, unif = unif, just_dropout = just_dropout))
            model.add(Dense(nb_classes, activation = 'softmax'))
            sgd = SGD(lr = 0.01, momentum = 0.9, decay = 0.00, nesterov = False)
            model.compile(loss='categorical_crossentropy',
                          optimizer= sgd,
                          metrics=['accuracy'])
            myfit = model.fit(small_X_train, small_Y_train, batch_size = batch_size, epochs = 150*D,
                              verbose = 1, validation_data=(X_test, Y_test), callbacks = [uncorrupted_train_history, optimizer_scheduler, CSVLogger('logs/lossp.{}.{}.{}.csv'.format(hbp, just_dropout, unif))])
            df = pd.DataFrame.from_dict(myfit.history)
            df['uncorrupted_train_acc'] = uncorrupted_train_history.accs
            df['uncorrupted_train_loss'] = uncorrupted_train_history.losses
            df['hbp'] = np.repeat(hbp, 150*D)
            df['uniform'] = np.repeat(unif, 150*D)
            df['just_dropout'] = np.repeat(just_dropout, 150*D)
            frames.append(df)
            out_frame = pd.concat(frames)
            out_frame.to_csv('./data/p_curves.csv')