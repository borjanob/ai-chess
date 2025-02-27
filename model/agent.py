from tensorflow.keras.layers import Input, Dense, Concatenate, Conv2D, Flatten, MaxPooling2D, Dropout, ReLU, BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MSE
from tensorflow.keras import Model
from tensorflow.nn import softmax
from tensorflow import reduce_mean, convert_to_tensor, squeeze, float32, GradientTape


class Agent(Model):

    """
    Defines a class for the actors used in reinforcement leraning where the states are represented as a 2-D image

    params:
    number_of_outputs: the number of outputs the neural net should return
    number_of_hidden_units: the number of hidden units in the neural net
    """

    def __init__(self,number_of_outputs,number_of_hidden_units):
        super(Agent,self).__init__()

        self.first_block = Sequential(
            [
                Conv2D(number_of_hidden_units, kernel_size=2, padding='same', strides=1, activation = 'relu',data_format = 'channels_last', kernel_initializer='he_normal'),
                Conv2D(number_of_hidden_units, kernel_size=2, padding='same', strides=1, activation = 'relu',data_format = 'channels_last', kernel_initializer='he_normal'),
                MaxPooling2D(pool_size=3, padding='same')
      
            ]
        )

        self.second_block = Sequential(
            [
                Conv2D(number_of_hidden_units, kernel_size=2, padding='same', strides=1, activation = 'relu', data_format = 'channels_last', kernel_initializer='he_normal'),

                MaxPooling2D(pool_size=3, padding='same')

            ]
        )

        self.prediction_block = Sequential(

            [
                Flatten(),
                Dense(128,activation = 'linear'),
                Dense(number_of_outputs, activation = 'linear')
            ]
        )

        self.relu = ReLU()

        self.dropout = Dropout(0.25)

        self.normalize = BatchNormalization()

    def call(self,data):
        x = self.first_block(data)
        x = self.normalize(x)
        x = self.second_block(x)
        x = self.normalize(x)

        x = self.prediction_block(x)

        return x