# Keras-2.0-CNN-and-CAE

Instructions:

On Python terminal do the following

>>> from CAE import trainCAE
>>> from CNN import runCNN

Use funtions: 

1> trainCAE()
'''
Parameters:
epochs - number of epochs for CAE
input_dim - number of samples to train CAE
show_encoded - show encoded representation or not
'''

2> runCNN()
'''
Parameters:
	epochs - enter number of epochs
	input_dim - number of samples you want to train the CNN on
	initialized - if you want to initialize weights from a CAE
	model_cae - file name from which you want to initialize CNN
'''
