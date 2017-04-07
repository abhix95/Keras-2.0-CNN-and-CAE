
# coding: utf-8



# In[1]:


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential
from keras import backend as K
from keras.models import Model


from keras.datasets import mnist
import numpy as np

from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt


# ### Import headers

# ### Loading data

# In[2]:



'''
Paramters:
epochs - number of epochs for CAE
input_dim - number of samples to train CAE
show_encoded - show encoded representation or not

'''

def trainCAE(epochs=20, input_dim=3000, show_encoded=True):


	(x_train, _), (x_test, _) = mnist.load_data()

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = np.reshape(x_train[:input_dim], (input_dim, 1, 28, 28)) 
	x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))  



	# In[3]:

	model = Sequential()

	model.add(Conv2D(8, (3, 3), activation='relu', padding='same',  input_shape=(1, 28, 28),name='conv1' ))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	print(model.layers[-1].output_shape)



	model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	print(model.layers[-1].output_shape)


	model.add(Conv2D(8, (3, 3), activation='relu', padding='same', name='conv3'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same', name='encoded'))
	print(model.layers[-1].output_shape)



	# at this point the representation is (4, 4, 8) i.e. 128-dimensional

	model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(size=(2, 2)))
	print(model.layers[-1].output_shape)


	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(size=(2, 2)))
	print(model.layers[-1].output_shape)


	model.add(Conv2D(8, (3, 3), activation='relu'))
	model.add(UpSampling2D(size=(2, 2)))
	print(model.layers[-1].output_shape)


	model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='decoded'))
	print(model.layers[-1].output_shape)

	model.compile(optimizer='adadelta', loss='binary_crossentropy')


	# In[ ]:


	model.fit(x_train, x_train,
	                epochs=epochs,
	                batch_size=128,
	                shuffle=True,
	                validation_data=(x_test, x_test),
	                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

	model.save('mnist_cae_'+str(epochs)+'e_'+str(input_dim)+'i.h5')
	# In[5]:




	if(show_encoded):
		decoded_imgs = model.predict(x_test)
		encoder = Model(inputs=model.input, outputs=model.get_layer('encoded').output)
		encoded_imgs = encoder.predict(x_test)



		n = 10
		plt.figure(figsize=(20, 4))
		for i in range(n):
		    # display original
		    ax = plt.subplot(2, n, i+1)
		    plt.imshow(x_test[i].reshape(28, 28))
		    plt.gray()
		    ax.get_xaxis().set_visible(False)
		    ax.get_yaxis().set_visible(False)

		    # display reconstruction
		    ax = plt.subplot(2, n, i + 1 + n)
		    plt.imshow(decoded_imgs[i].reshape(28, 28))
		    plt.gray()
		    ax.get_xaxis().set_visible(False)
		    ax.get_yaxis().set_visible(False)
		plt.show()


		# In[ ]:

		n = 10
		plt.figure(figsize=(20, 8))
		for i in range(n):
		    ax = plt.subplot(1, n, i+1)
		    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
		    plt.gray()
		    ax.get_xaxis().set_visible(False)
		    ax.get_yaxis().set_visible(False)
		plt.show()


	# In[ ]:



