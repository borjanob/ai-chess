from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, ReLU, BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import Model



class Agent(Model):

    """
    Defines a class for the actors used in reinforcement leraning where the states are represented as a 2-D image

    params:
    number_of_outputs: the number of outputs the neural net should return
    number_of_hidden_units: the number of hidden units in the neural net
    """

    def __init__(self,number_of_outputs: int,number_of_hidden_units: int, **kwargs):
        super(Agent,self).__init__(**kwargs)

        self.number_of_outputs = number_of_outputs

        self.number_of_hidden_units = number_of_hidden_units

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
    

    def get_config(self):
        base_config = super().get_config()

        config = {
            "number_of_outputs": self.number_of_outputs,
            "number_of_hidden_units" :self.number_of_hidden_units
        }
        
        return {**base_config, **config} 