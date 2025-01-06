import numpy as np
import random
from collections import deque
from tensorflow.keras.layers import Input, Dense, Concatenate,Flatten,Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MSE
from tensorflow import reduce_mean, convert_to_tensor, squeeze, GradientTape, float32
import tensorflow


class ReduceMeanLayer(Layer):
    def call(self, inputs):
        return reduce_mean(inputs, axis=1, keepdims=True)



class DuelingDQN:
    def __init__(self, state_space_shape, num_actions,layers, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        """
        Initializes Dueling Deep Q Network agent.
        :param state_space_shape: shape of the observation space
        :param num_actions: number of actions
        :param learning_rate: learning rate
        :param discount_factor: discount factor
        :param batch_size: batch size
        :param memory_size: maximum size of the experience replay memory
        """
        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.layers = layers
        self.build_model(layers)

    def _build_model(self, layers):
        """
        Builds a model with the given layers.
        :param layers: layers for the model
        """
        input_layer = Input(shape=self.state_space_shape)

        x = input_layer
        for layer in layers:
            x = layer(x)

        v = Dense(1)(x)
        a = Dense(self.num_actions)(x)

        q = v + (a - ReduceMeanLayer()(a))


        model = Model(inputs=input_layer, outputs=q)
        model.compile(Adam(self.learning_rate), loss=MeanSquaredError())
        return model

    def build_model(self, layers):
        """
        Builds the main and target network with the given layers.
        :param layers: layers for the models
        """
        self.model = self._build_model(layers)
        self.target_model = self._build_model(layers)
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        """
        Adds experience tuple to experience replay memory.
        :param state: current state
        :param action: performed action
        :param reward: reward received for performing action
        :param next_state: next state
        :param done: if episode has terminated after performing the action in the current state
        """
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        """
        Synchronize the target model with the main model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, epsilon, action_mask):
        """
        Returns the best action following epsilon greedy policy for the current state.
        :param state: current state
        :param epsilon: exploration rate
        :return:
        """

        probability = np.random.random() + epsilon / self.num_actions
         # AKO E EXPLORATION
        if probability < epsilon:

            # SELECT RANDOM MOVE UNTIL WE GET A VALID RADOM MOVE FROM ACTION MASK
            is_valid_action = False
            # safety so loop doesnt get stuck
            exit_counter = 0
            while(is_valid_action == False and exit_counter<=1000):
                action_number = np.random.randint(0, self.num_actions)
                exit_counter += 1
                if action_mask[action_number] == 1:
                    is_valid_action = True

            return action_number

        else:
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)

            # SE MNOZI SO ACTION MASK ZA DA SE NAPRAT 0 TIE AKCII SO NE SE LEGALNI
            # A DA SI OSTANAT x1 TIE SO SE LEGAL

            full_predictions = self.model.predict(state)[0]

            # MULTIPLY WITH ACTION MASK
            legal_moves = [a*b for a,b in zip(full_predictions,action_mask)]
            
            legal_moves[legal_moves == 0.0] = -np.inf

            return np.argmax(legal_moves)



    def load(self, path_to_weights):
        """
        Loads the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.load_weights(path_to_weights)

    def save(self, model_name, episode):
        """
        Stores the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.save_weights(f'duelingdqn_{model_name}_{episode}.h5')

    def train(self):
        """
        Performs one step of model training.
        """
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        if isinstance(self.state_space_shape, tuple):
            states = np.zeros((batch_size,) + self.state_space_shape)
        else:
            states = np.zeros((batch_size, self.state_space_shape))
        actions = np.zeros((batch_size, self.num_actions))

        for i in range(len(minibatch)):
            state, action, reward, next_state, done = minibatch[i]
            if done:
                max_future_q = reward
            else:
                if isinstance(self.state_space_shape, tuple):
                    next_state = next_state.reshape((1,) + self.state_space_shape)
                else:
                    next_state = next_state.reshape(1, self.state_space_shape)
                    
                next_state = np.array(next_state, dtype=np.float32)
                max_future_q = (reward + self.discount_factor *
                                np.amax(self.target_model.predict(next_state)[0]))
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)

            target_q = self.model.predict(state)[0]
            target_q[action] = max_future_q
            states[i] = state
            actions[i] = target_q

        self.model.train_on_batch(states, actions)
