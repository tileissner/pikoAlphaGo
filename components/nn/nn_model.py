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


		# predefined seed
		initializer = tf.keras.initializers.RandomNormal(stddev=0.05, mean=0, seed=8)
		#initializer = tf.keras.initializers.Zeros()
		#initializer = tf.keras.initializers.Ones()


		self.conv1 = Conv2D(filters=32, kernel_size=2, name='conv1', kernel_initializer=initializer)
		self.normalization1 = BatchNormalization()
		self.activation1 = Activation(activation='relu', name='act1')
		self.conv2 = Conv2D(filters=32, kernel_size=2, name="conv2", kernel_initializer=initializer)
		self.normalization2 = BatchNormalization()
		self.activation2 = Activation(activation='relu', name="act2")
		self.conv3 = Conv2D(filters=32, kernel_size=2, name="conv3", kernel_initializer=initializer)
		self.normalization3 = BatchNormalization()
		self.activation3 = Activation(activation='relu', name='act3')

		"""
		self.conv4 = Conv2D(filters=32, kernel_size=2, name='conv4')
		#normalization4 = BatchNormalization(conv4)
		self.activation4 = Activation(activation='relu', name='act4')
		self.conv5 = Conv2D(filters=32, kernel_size=2, name='conv5')
		#normalization5 = BatchNormalization(conv5)
		self.activation5 = Activation(activation='relu', name='act5')
		"""

		"""value head"""
		self.vh_conv = Conv2D(filters=1, kernel_size=1, name='vh_conv', kernel_initializer=initializer)
		self.vh_norm = BatchNormalization(name='vh_norm')
		self.vh_activation = Activation(activation='relu', name='vh_act')
		self.vh_flatten = Flatten(name='vh_flat')
		"""
		wurde im original paper von 361 = 19x19 auf 256 connected
		bei uns ist feld aber deutlich kleiner 5x5 = 25 also auf 16 vllt?
		"""
		self.vh_dense1 = Dense(units=16, name='vh_fc1', kernel_initializer=initializer)		#Units = ?
		self.vh_dense2 = Dense(units=1, activation='tanh', name='vh_out', kernel_initializer=initializer)

		"""policy head"""
		self.ph_conv = Conv2D(filters=2, kernel_size=1, name='ph_conv',kernel_initializer=initializer)
		self.ph_norm = BatchNormalization(name='ph_norm')
		self.ph_activation = Activation(activation='relu', name='ph_act')
		self.ph_flatten = Flatten(name='ph_flat')
		self.ph_dense = Dense(units=26, name='ph_out', activation=tf.nn.softmax, kernel_initializer=initializer) #Units = Boardsize + 1 = N**2 + 1

	def call(self, inputs):  # create model
		x = self.conv1(inputs)
		x = self.normalization1(x)
		x = self.activation1(x)
		x = self.conv2(x)
		x = self.normalization2(x)
		x = self.activation2(x)
		x = self.conv3(x)
		x = self.normalization3(x)
		x = self.activation3(x)

		value = self.vh_conv(x)
		value = self.vh_norm(value)
		value = self.vh_activation(value)
		value = self.vh_flatten(value)
		value = self.vh_dense1(value)
		value = self.vh_dense2(value)

		policy = self.ph_conv(x)
		policy = self.ph_norm(policy)
		policy = self.ph_activation(policy)
		policy = self.ph_flatten(policy)
		policy = self.ph_dense(policy)

		return value, policy
