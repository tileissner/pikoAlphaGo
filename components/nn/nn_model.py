import tensorflow.keras
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *


def create_model():
	aliases = {}
	Input_1 = Input(shape=(28, 28, 1), name='Input_1')
	Conv2D_3 = Conv2D(name='Conv2D_3',filters= 32,kernel_size= 2)(Input_1)
	BatchNormalization_1 = BatchNormalization(name='BatchNormalization_1')(Conv2D_3)
	Activation_1 = Activation(name='Activation_1',activation= 'relu' )(BatchNormalization_1)
	Conv2D_4 = Conv2D(name='Conv2D_4',filters= 32,kernel_size= 2)(Activation_1)
	BatchNormalization_2 = BatchNormalization(name='BatchNormalization_2')(Conv2D_4)
	Activation_2 = Activation(name='Activation_2',activation= 'relu' )(BatchNormalization_2)
	Conv2D_5 = Conv2D(name='Conv2D_5',filters= 32,kernel_size= 2)(Activation_2)
	BatchNormalization_3 = BatchNormalization(name='BatchNormalization_3')(Conv2D_5)
	Activation_3 = Activation(name='Activation_3',activation= 'relu' )(BatchNormalization_3)
	Conv2D_6 = Conv2D(name='Conv2D_6',filters= 32,kernel_size= 2)(Activation_3)
	BatchNormalization_4 = BatchNormalization(name='BatchNormalization_4')(Conv2D_6)
	Activation_4 = Activation(name='Activation_4',activation= 'relu' )(BatchNormalization_4)
	Conv2D_7 = Conv2D(name='Conv2D_7',filters= 32,kernel_size= 2)(Activation_4)
	BatchNormalization_5 = BatchNormalization(name='BatchNormalization_5')(Conv2D_7)
	Activation_5 = Activation(name='Activation_5',activation= 'relu' )(BatchNormalization_5)
	Conv2D_9 = Conv2D(name='Conv2D_9',filters= 12,kernel_size= 1)(Activation_5)
	BatchNormalization_7 = BatchNormalization(name='BatchNormalization_7')(Conv2D_9)
	Activation_8 = Activation(name='Activation_8',activation= 'relu' )(BatchNormalization_7)
	Flatten_2 = Flatten(name='Flatten_2')(Activation_8)
	Dense_2 = Dense(name='Dense_2',units= 256)(Flatten_2)
	Dense_3 = Dense(name='Dense_3',units= 1,activation= 'tanh' )(Dense_2)
	Conv2D_8 = Conv2D(name='Conv2D_8',filters= 2,kernel_size= 1)(Activation_5)
	BatchNormalization_6 = BatchNormalization(name='BatchNormalization_6')(Conv2D_8)
	Activation_7 = Activation(name='Activation_7',activation= 'relu' )(BatchNormalization_6)
	Flatten_1 = Flatten(name='Flatten_1')(Activation_7)
	Dense_18 = Dense(name='Dense_18',units= 362)(Flatten_1)

	model = Model([Input_1], [Dense_3, Dense_18])
	return aliases, model

def get_optimizer():
	return SGD()

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'categorical_crossentropy'

def get_batch_size():
	return 32

def get_num_epoch():
	return 10

"""
def get_data_config():
	return '{"mapping": {"Digit Label": {"type": "Categorical", "port": "OutputPort0", "shape": "", "options": {}}, "Image": {"type": "Image", "port": "InputPort0", "shape": "", "options": {"pretrained": "None", "Augmentation": false, "rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "shear_range": 0, "horizontal_flip": false, "vertical_flip": false, "Scaling": 1, "Normalization": false, "Resize": false, "Width": 28, "Height": 28}}}, "numPorts": 1, "samples": {"training": 56000, "validation": 14000, "test": 0, "split": 1}, "dataset": {"name": "mnist", "type": "public", "samples": 70000}, "datasetLoadOption": "batch", "shuffle": true, "kfold": 1}'
"""