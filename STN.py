from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from get_initial_weights import get_initial_weights
from BilinearInterpolation import BilinearInterpolation
import numpy as np


class STNMethod(Layer):
    def __init__(self, name, **kwargs):
        super(STNMethod, self).__init__(name=name)
        super(STNMethod, self).__init__(**kwargs)

    def build(self, input_shape):
        b, w, h, c = input_shape
        self.sampling_size = (w, w)
        self.mp1 = MaxPool2D(pool_size=(2, 2))
        self.conv1 = Conv2D(20, kernel_size=5, padding="same")
        self.mp2 = MaxPool2D(pool_size=(2, 2))
        self.conv2 = Conv2D(20, kernel_size=5, padding="same")
        self.flatten = Flatten()
        self.dense1 = Dense(50, activation="relu")
        self.weight = self.get_initial_weights(50)
        self.dense2 = Dense(6, weights=self.weight)
        self.bi = BilinearInterpolation(self.sampling_size)
        # self.bi = layers.Resizing(height=h, width=w)
        self.reshape = layers.Reshape((w, h, c), name=f'Reshape_None_1_1{c}')
        super().build(input_shape)

    def get_initial_weights(self, output_size):
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        W = np.zeros((output_size, 6), dtype='float32')
        weights = [W, b.flatten()]
        return weights

    def call(self, inputs,  *args, **kwargs):
        x = self.mp1(inputs)
        x = self.conv1(x)
        x = self.mp2(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.bi([inputs, x])
        x = self.reshape(x)
        print("Shape", x.get_shape().as_list())
        return x

    def get_config(self):
        base_config = super(STNMethod, self).get_config()
        return dict(list(base_config.items()) )





