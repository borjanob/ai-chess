from tensorflow.keras.layers import Input, Dense, Concatenate, Conv2D, Flatten, MaxPooling2D, Dropout, ReLU
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
                Conv2D(number_of_hidden_units, kernel_size=3, padding='same', strides=1, activation = 'relu',data_format = 'channels_last'),
                Conv2D(number_of_hidden_units, kernel_size=3, padding='same', strides=1, activation = 'relu',data_format = 'channels_last'),
                MaxPooling2D(pool_size=3, padding='same')
                #Dropout(0.25)
            ]
        )

        self.second_block = Sequential(
            [
                Conv2D(number_of_hidden_units, kernel_size=3, padding='same', strides=1, activation = 'relu', data_format = 'channels_last'),
                Conv2D(number_of_hidden_units, kernel_size=3, padding='same', strides=1, activation = 'relu',),
                MaxPooling2D(pool_size=3, padding='same')
            #Dropout(0.25)
            ]
        )

        self.third_block = Sequential(
            [
                Conv2D(number_of_hidden_units, kernel_size=3, padding='same', strides=1, activation = 'relu'),
                Conv2D(number_of_hidden_units, kernel_size=3, padding='same', strides=1, activation = 'relu'),
                MaxPooling2D(pool_size=3, padding='same')
            ]
        )

        self.prediction_block = Sequential(

            [
                Flatten(),
                Dense(128,activation = 'relu'),
                #Dropout(0.25),
                Dense(64,activation = 'relu'),

                
                Dense(number_of_outputs, activation = 'linear')
            ]
        )

        self.relu = ReLU()

        self.dropout = Dropout(0.25)

    def call(self,data):
        x = self.first_block(data)
        x = self.second_block(x)
        x = self.third_block(x)
        x = self.prediction_block(x)

        #output probabilities
        #x = softmax(x)
        return x