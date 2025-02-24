import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten, Dropout, ReLU, BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MSE
from tensorflow.keras import Model
from tensorflow.nn import softmax
from tensorflow import reduce_mean, convert_to_tensor, squeeze, float32, GradientTape


class Actor(tf.keras.Model):
    def __init__(self, input_shape, action_dim, num_of_hidden_units = 128):
        super(Actor, self).__init__()
        # Convolutional layers for processing 8x8x111 input
        self.conv1 = tf.keras.layers.Conv2D(num_of_hidden_units, 3, strides=2, activation='relu', input_shape=input_shape)

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
        
        # Output layer for action logits
        self.logits = tf.keras.layers.Dense(action_dim)  # No activation, outputs raw logits

    def call(self, state):
        # Forward pass
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        logits = self.logits(x) 
        return tf.nn.softmax(logits) # Outputs logits for categorical distribution
