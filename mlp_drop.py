import numpy as np
np.random.seed(42)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import LearningRateScheduler, CSVLogger


def schedule(x):
     x = np.array(x, dtype = 'float32')
     lr = np.piecewise(x,
                       [x <= 25,
                        (x > 25) & (x <= 50),
                        (x > 50) & (x <= 75),
                        x > 75],
                       [0.1,
                        0.1 * 0.2,
                        0.1 * 0.2 ** 2,
                        0.1 * 0.2 ** 3])
     return(float(lr))

optimizer_schedule = LearningRateScheduler(schedule)

batch_size = 512
nb_classes = 10
wd = 0.00001

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(X_train_s, y_train_s), (X_test, y_test) = mnist.load_data()

X_train_s = X_train_s.astype('float32')
X_test = X_test.astype('float32')
X_train_s /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train_s = np_utils.to_categorical(y_train_s, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Flatten(input_shape = (img_rows, img_cols)))
model.add(Dropout(0.25))
model.add(Dense(8192, activation = 'relu', kernel_regularizer = regularizers.l2(wd)))
model.add(Dropout(0.5))
model.add(Dense(8192, activation = 'relu', kernel_regularizer = regularizers.l2(wd)))
model.add(Dense(nb_classes, activation = 'softmax',
               kernel_regularizer = regularizers.l2(wd)))
sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0.00, nesterov = False)
model.compile(loss = 'categorical_crossentropy',
             optimizer = sgd,
             metrics = ['accuracy'])
fit = model.fit(X_train_s, Y_train_s, batch_size = batch_size, epochs = 100,
               verbose = 1, validation_data = (X_test, Y_test),
               callbacks = [optimizer_schedule, CSVLogger('logs/mlp_drop.csv')])

print(format(100 * (1 - fit.history['val_acc'][-1]), '.2f'))