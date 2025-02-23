import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Flatten, Dropout, ReLU, BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MSE
from tensorflow.keras import Model
from tensorflow.nn import softmax
from tensorflow import reduce_mean, convert_to_tensor, squeeze, float32, GradientTape


ACTION_EPS = 1e-6  # Small value to prevent log(0) issues

class Actor(Model):
    def __init__(self, state_dim, action_dim,num_of_hidden_units):
        super(Actor, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.num_of_hidden_units = num_of_hidden_units

        self.conv_block = Sequential(
            [
                Conv2D(num_of_hidden_units, kernel_size=2, padding='same', strides=1, activation = 'relu',data_format = 'channels_last', kernel_initializer='he_normal'),
                Conv2D(num_of_hidden_units, kernel_size=2, padding='same', strides=1, activation = 'relu',data_format = 'channels_last', kernel_initializer='he_normal'),
                MaxPooling2D(pool_size=3, padding='same')
      
            ]
        )

        self.conv_block2 = Sequential(
            [
                Conv2D(num_of_hidden_units, kernel_size=2, padding='same', strides=1, activation = 'relu', data_format = 'channels_last', kernel_initializer='he_normal'),

                MaxPooling2D(pool_size=3, padding='same'),
                Flatten()

            ]
        )


        self.fc1_actor = tf.keras.layers.Dense(num_of_hidden_units, activation='relu')
        self.fc2_actor = tf.keras.layers.Dense(num_of_hidden_units, activation='relu')
        self.conv1_actor = tf.keras.layers.Dense(num_of_hidden_units, activation='relu')
        self.conv2_actor = tf.keras.layers.Dense(num_of_hidden_units, activation='relu')
        self.conv3_actor = tf.keras.layers.Dense(num_of_hidden_units, activation='relu')
        self.fc3_actor = tf.keras.layers.Dense(num_of_hidden_units, activation='relu')
        self.fc4_actor = tf.keras.layers.Dense(num_of_hidden_units, activation='relu')
        self.pi_head = tf.keras.layers.Dense(action_dim, activation=None)


    def call(self, inputs, training=False):

        inputs = self.conv_block(inputs)
        inputs = self.conv_block2(inputs)
        split_0 = self.fc1_actor(inputs[:, 0:1, -1])
        split_1 = self.fc2_actor(inputs[:, 1:2, -1])
        split_2 = self.conv1_actor(inputs[:, 2:3, :])
        split_2 = tf.reshape(split_2, [-1, self.num_of_hidden_units])
        split_3 = self.conv2_actor(inputs[:, 3:4, :])
        split_3 = tf.reshape(split_3, [-1, self.num_of_hidden_units])
        split_4 = self.conv3_actor(inputs[:, 4:5, :self.a_dim])
        split_4 = tf.reshape(split_4, [-1, self.num_of_hidden_units])
        split_5 = self.fc3_actor(inputs[:, 5:6, -1])

        merge_net = tf.concat([split_0, split_1, split_2, split_3, split_4, split_5], axis=1)

        pi_net = self.fc4_actor(merge_net)
        pi_net = tf.nn.relu(pi_net)
        pi = tf.nn.softmax(self.pi_head(pi_net), axis=-1)
        pi = tf.clip_by_value(pi, ACTION_EPS, 1. - ACTION_EPS)
        return pi
