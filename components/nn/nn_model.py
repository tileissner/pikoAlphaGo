import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *


class NeuralNetwork(Model):

	def __init__(self):
		super(NeuralNetwork, self).__init__()
		#TODO input shape fixen
		self.input_ = Input(shape=())
		self.conv1 = Conv2D(filters=32, kernel_size=2)(self.input_)
		self.normalization1 = BatchNormalization(self.conv1)
		self.activation1 = Activation(activation='relu')(self.normalization1)
		self.conv2 = Conv2D(filters=32, kernel_size=2)(self.activation1)
		self.normalization2 = BatchNormalization(self.conv2)
		self.activation2 = Activation(activation='relu')(self.normalization2)
		self.conv3 = Conv2D(filters=32, kernel_size=2)(self.activation2)
		self.normalization3 = BatchNormalization(self.conv3)
		self.activation3 = Activation(activation='relu')(self.normalization3)
		self.conv4 = Conv2D(filters=32, kernel_size=2)(self.activation3)
		self.normalization4 = BatchNormalization(self.conv4)
		self.activation4 = Activation(activation='relu')(self.normalization4)
		self.conv5 = Conv2D(filters=32, kernel_size=2)(self.activation4)
		self.normalization5 = BatchNormalization(self.conv5)
		self.activation5 = Activation(activation='relu')(self.normalization5)

		#value head
		self.vh_conv = Conv2D(filters=1, kernel_size=1)(self.activation5)
		self.vh_norm = BatchNormalization()(self.vh_conv)
		self.vh_activation = Activation(activation='relu')(self.vh_norm)
		self.vh_flatten = Flatten()(self.vh_activation)
		self.vh_dense1 = Dense(units=256)(self.vh_flatten)
		self.vh_dense2 = Dense(units=1, activation='tanh')(self.vh_dense1)

		#policy head
		self.ph_conv = Conv2D(filters=2, kernel_size=1)(self.activation5)
		self.ph_norm = BatchNormalization()(self.ph_conv)
		self.ph_activation = Activation(activation='relu')(self.ph_norm)
		self.ph_flatten = Flatten()(self.ph_activation)
		self.ph_dense = Dense(units=362)(self.ph_flatten)

		#https://www.tensorflow.org/api_docs/python/tf/keras/Model
		return Model(inputs=[self.input_], outputs=[self.vh_dense2, self.ph_dense])
	
model = NeuralNetwork()
model.summary()