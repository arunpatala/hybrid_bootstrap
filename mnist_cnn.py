import numpy as np
np.random.seed(42)
from keras import backend as K
from theano import tensor as T
from keras.engine.topology import Layer
from keras.initializers import Constant
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.metrics import log_loss
from keras.callbacks import Callback
from keras.layers import Input


# We do need to redefine HB so we can't use noweb here
def hybo(x, p, shift, seed = None, unif = True, just_dropout = False):
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

  mask = rng.binomial(x.shape, p = retain_prob, dtype = x.dtype)
  if just_dropout:
      x = x * mask / retain_prob
  else:
      x = x * mask + (1 - mask) * T.roll(x, shift = shift, axis = 0)

  return (x, retain_prob)



class HB(Layer):
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
      self.last_retain_prob = None
      super(HB, self).__init__(**kwargs)

  def build(self, input_shape):
      self.p = self.add_weight(shape=(),
                               name = 'p',
                               initializer=Constant(value = self.init_p),
                               trainable=False)
      super(HB, self).build(input_shape)

  def call(self, x, mask=None):
      if 0. < self.p.get_value() < 1.:
          x, self.last_retain_prob = hybo(x, p = self.p, shift = self.shift,
                                      unif = self.unif,
                                      just_dropout = self.just_dropout)
      return x

  def get_config(self):
      config = {'init_p': self.init_p, 'p': self.p, 'shift': self.shift,
                'unif': self.unif, 'just_dropout': self.just_dropout}
      base_config = super(HB, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))




class RecordGrads(Callback):
  '''Callback to keep track of uncorrupted training losses'''
  def __init__(self, data, num_samples, sample_batch_size):
      self.data_X, self.data_Y = data
      self.num_samples = num_samples
      self.sample_batch_size = sample_batch_size

  def on_train_begin(self, logs={}):
      gradients = self.model.optimizer.get_gradients(self.model.total_loss, self.model.trainable_weights)
      inputs = [self.model.inputs[0],
                self.model.sample_weights[0],
                self.model.targets[0]]
      last_retain_probs = []
      for layer in self.model.layers:
          if layer.name[0:2] == 'hb':
              last_retain_probs.append(layer.last_retain_prob)
      gradients_and_ps = gradients + last_retain_probs
      self.get_gradients_and_ps = K.function(inputs = inputs, outputs = gradients_and_ps)
      self.num_lay = len(self.model.trainable_weights) / 2
      self.sampled_gradients_list = []
      self.just_first_point = np.zeros(self.sample_batch_size)
      self.just_first_point[0] = 1

  def on_epoch_end(self, batch, logs={}):
      sampled_gradients = np.zeros((self.num_samples * self.num_lay, 2))
      for i in range(self.num_samples):
          random_sample = np.random.randint(self.data_X.shape[0], size = self.sample_batch_size)
          inputs = [self.data_X[random_sample], self.just_first_point, self.data_Y[random_sample]]
          gsandps = self.get_gradients_and_ps(inputs)
          for j in range(self.num_lay):
              weight_vector = np.concatenate([np.array(gsandps[2 * j]).flatten(), np.array(gsandps[2 * j + 1]).flatten()])
              norm = np.linalg.norm(weight_vector)
              sampled_gradients[i * self.num_lay + j, 0] = norm
              sampled_gradients[i * self.num_lay + j, 1] = 1 - gsandps[2 * self.num_lay + j][0]
      self.sampled_gradients_list.append(sampled_gradients)




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
  lr = np.piecewise(x, [x < 500, (x >= 500) & (x < 1000), x >= 1000],
                      [0.01, 0.001, 0.0001])
  momentum = np.piecewise(x, [x < 500, x >= 500],
                      [0.9, 0.99])
  return((lr, momentum))

frames = []
shuffle_perm = np.arange(small_X_train.shape[0])
np.random.shuffle(shuffle_perm)
record_grads = RecordGrads([small_X_train[shuffle_perm], small_Y_train[shuffle_perm]], num_samples = 100, sample_batch_size = 128)
optimizer_scheduler = OptimizerScheduler(schedule)
just_dropout = False
unif = True
hbp = 0.5

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
myfit = model.fit(small_X_train, small_Y_train, batch_size = batch_size, epochs = 1500,
                verbose = 1, validation_data=(X_test, Y_test), callbacks = [record_grads, optimizer_scheduler])

grads_and_ps = np.concatenate(record_grads.sampled_gradients_list)
grads_and_ps = pd.DataFrame(grads_and_ps)
grads_and_ps.to_csv("./data/grads_and_ps.csv")