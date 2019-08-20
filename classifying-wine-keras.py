#https://github.com/luiseduardogfranca/classifying-wine

import numpy as np
from data_proccesing import DataProcessing
from keras.models import Sequential
from keras.layers import Dense 
from keras.utils import to_categorical
from keras.optimizers import Adam 
from keras. initializers import TruncatedNormal 
import matplotlib.pyplot as plt


#data processing 

dp = DataProcessing()
dp.processing('dataset/wine.data.txt', 14)

X_train, X_test = np.load('dataset/X_train.npy'), np.load('dataset/X_test.npy')
Y_train, Y_test = np.load('dataset/Y_train.npy'), np.load('dataset/Y_test.npy')


#binary array
Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)

#for model
input_dim = len(X_train[0, :])
class_num = len(Y_train[0, :]) 

model = Sequential()
init = TruncatedNormal(stddev=0.01, seed=10)

#config model 
model.add(Dense(units=50, activation='relu', input_dim=input_dim, kernel_initializer=init))
model.add(Dense(units=class_num, activation='softmax', kernel_initializer=init))

#optimizer We are using adam optimizer
#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adam = Adam(lr=0.007)

#compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#fit
#fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None,
# shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)
history = model.fit(X_train, Y_train, epochs=1000, validation_data=(X_test, Y_test), shuffle=False, verbose=0)
#A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs,
# as well as validation loss values and validation metrics values (if applicable).

plt.subplot(2, 2, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()