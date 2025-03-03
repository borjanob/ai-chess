import numpy as np
import random
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow import float32
import tensorflow as tf

class DDQN:
    def __init__(self, state_space_shape, num_actions, model, target_model, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        """
        Initializes Double Deep Q Network agent.
        :param state_space_shape: shape of the observation space
        :param num_actions: number of actions
        :param model: Keras model
        :param target_model: Keras model
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
        self.model = model
        self.target_model = target_model
        #self.update_target_model()

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


    def _get_legal_moves(self,predictions,action_mask):
        legal_moves = []

        for i in range(len(predictions)):
            if action_mask[i] == 0:
                legal_moves.append(-np.inf)
            else:
                legal_moves.append(predictions[i])
        return legal_moves


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
        :param action_mask: legal moves for current state
        :return:
        """

        probability = np.random.random() + epsilon / self.num_actions
         # AKO E EXPLORATION
        if probability < epsilon:

            # SELECT RANDOM MOVE UNTIL WE GET A VALID RADOM MOVE FROM ACTION MASK
            is_valid_action = False
            # safety so loop doesnt get stuck
            exit_counter = 0
            while(is_valid_action == False and exit_counter<=3000):
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
            # legal_moves = [a*b for a,b in zip(full_predictions,action_mask)]

            # final_moves = [x if x != 0.0 or x != -0.0 else -np.inf for x in legal_moves]
            legal_moves = self._get_legal_moves(full_predictions,action_mask)
            print(max(legal_moves))
            return np.argmax(legal_moves)


    def load(self, path_to_weights, update_target_model = True):
        """
        Loads the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """

        zeros = tf.zeros((1,8,8,111), dtype=float32)
        self.model.predict(zeros)

        self.model.load_weights(path_to_weights)

        if update_target_model:
            self.target_model.predict(zeros)
            self.target_model.load_weights(path_to_weights)

    def load_full_model(self, path_to_model):

        self.model = load_model(path_to_model)


    def save_full_model(self, episode):
        self.model.save(f'ddqn_model_{episode}', save_format = 'tf')

    def save(self, model_name, episode):
        """
        Stores the weights of the model at specified episode checkpoint.
        :param model_name: name of the model
        :param episode: episode checkpoint
        """
        self.model.save_weights(f'ddqn_{model_name}_{episode}.weights.h5')

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
                
                max_action = np.argmax(self.model.predict(next_state)[0])
                max_future_q = (reward + self.discount_factor *
                                self.target_model.predict(next_state)[0][max_action])
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)
            target_q = self.model.predict(state)[0]
            target_q[action] = max_future_q
            states[i] = state
            actions[i] = target_q

        self.model.train_on_batch(states, actions)