import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten
from tensorflow.keras.models import Sequential


class Critic(tf.keras.Model):
    def __init__(self, input_shape: int, num_of_hidden_units = 128):
        super(Critic, self).__init__()

        self.input_shp = input_shape
        self.num_of_hidden_units = num_of_hidden_units

        self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=2, activation='relu', input_shape=input_shape)

        self.conv2 = Sequential(
            [
                Conv2D(num_of_hidden_units, kernel_size=2, padding='same', strides=1, activation = 'relu',data_format = 'channels_last', kernel_initializer='he_normal'),
                Conv2D(num_of_hidden_units, kernel_size=2, padding='same', strides=1, activation = 'relu',data_format = 'channels_last', kernel_initializer='he_normal'),
                MaxPooling2D(pool_size=3, padding='same')
      
            ]
        )

        self.conv3 = Sequential(
            [
                Conv2D(num_of_hidden_units, kernel_size=2, padding='same', strides=1, activation = 'relu', data_format = 'channels_last', kernel_initializer='he_normal'),

                MaxPooling2D(pool_size=3, padding='same'),
                Flatten()

            ]
        )


        #self.conv2 = tf.keras.layers.Conv2D(64, 2, strides=1, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.value(x)
    

    def get_config(self):

        base = super().get_config()

        config = {
            
            "input_shape": self.input_shp,
            "num_of_hidden_units": self.num_of_hidden_units

        }

        return {**base, **config}