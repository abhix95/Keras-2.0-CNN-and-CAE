
# coding: utf-8

# ### Aprroaches explored:
# #### 1. Simple CNN
# #### 2. CAE to initialize CNN weights
# #### 3. Fine-tune a VGG net

# In[1]:


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.models import Sequential
from keras import backend as K
from keras.models import Model

from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np

from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt


# ### Import headers

# ### Loading data

# In[2]:


'''
Parameters:
	epochs - enter number of epochs
	input_dim - number of samples you want to train the CNN on
	initialized - if you want to initialize weights from a CAE
	model_cae - file name from which you want to initialize CNN
'''


def runCNN(epochs=20, input_dim=1000, initialized=True, model_cae='mnist_cae_20e_3000i.h5'):


	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.

	x_train = np.reshape(x_train[10000:10000+input_dim], (input_dim, 1, 28, 28)) 
	x_test = np.reshape(x_test[0:500], (500, 1, 28, 28))  

	y_train = y_train[10000:10000+input_dim]
	y_test = y_test[0:500]

	y_train = to_categorical(y_train, num_classes=10)
	y_test = to_categorical(y_test, num_classes=10)


	# In[3]:

	model = Sequential()

	model.add(Conv2D(8, (3, 3), activation='relu', padding='same',  input_shape=(1, 28, 28), name='conv1' ))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

	model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

	model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv3'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same', name='encoded'))

	model.add(Flatten())
	model.add(Dense(16))

	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(10))
	model.add(Activation('softmax'))
	
	if(initialized):
		model.load_weights(model_cae, by_name=True)
		name = 'mnist_initializedCNN_'+str(epochs)+'e_'+str(input_dim)+'i.h5'
	else:
		name = 'mnist_CNN_'+str(epochs)+'e_'+str(input_dim)+'i.h5'


	model.compile(optimizer='adadelta', loss='categorical_crossentropy',metrics=['accuracy'])


	# In[ ]:


	model.fit(x_train, y_train,
	                epochs=epochs,
	                batch_size=128,
	                shuffle=True)


	model.save(name)
	# In[5]:
	print(model.evaluate(x_test,y_test))
	return(name)


	# In[ ]:



