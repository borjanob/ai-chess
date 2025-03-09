import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
from collections import deque
import random
from tensorflow.keras.models import load_model
from model.ppo_actor import Actor
from model.ppo_critic import Critic

class PPO:
    def __init__(self, input_shape, number_of_actions,
                 clip_ratio=0.2, discount_factor=0.99, lam=0.95,
                 actor_lr=3e-4, critic_lr=1e-3, 
                 batch_size=64, epochs=4, memory_size = 100, 
                 num_of_hidden_units_actor = 128, num_of_hidden_units_critic = 128):
        
    
        self.actor = Actor(input_shape, number_of_actions, num_of_hidden_units_actor)
        self.critic = Critic(input_shape, num_of_hidden_units_critic)
        
        self.state_space_shape = input_shape
        self.num_actions = number_of_actions

        # Optimizers i vidi dali treba clipnorm ovde ili nema potreba
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr, clipnorm=2)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr,clipnorm=2)
        
  
        self.memory = deque(maxlen=memory_size)
 
        self.clip_ratio = clip_ratio
        self.discount_factor = discount_factor
        self.lam = lam
        self.batch_size = batch_size
        self.epochs = epochs
        self.entropy_coef = 0.01
        self.value_coef = 0.5
    
        self.input_shape = input_shape
        self.action_dim = number_of_actions

    def update_memory(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def compute_advantages(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.discount_factor * next_value * mask - values[t]
            advantages[t] = delta + self.discount_factor * self.lam * mask * last_advantage
            last_advantage = advantages[t]
            next_value = values[t]
            
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages
    


    def _get_legal_moves(self,predictions,action_mask):
        legal_moves = []

        for i in range(len(predictions)):
            if action_mask[i] == 0:
                legal_moves.append(-np.inf)
            else:
                legal_moves.append(predictions[i])
        return legal_moves


    def get_action(self, state, epsilon, action_mask, explore= True):
        """
        Returns the best action following epsilon greedy policy for the current state.
        :param state: current state
        :param epsilon: exploration rate
        :return:
        """
        probability = np.random.random() + epsilon / self.num_actions
        full_predictions = self.actor.predict(state)[0]

        # AKO E EXPLORATION
        if probability < epsilon and explore:

            # SELECT RANDOM MOVE UNTIL WE GET A VALID RADOM MOVE FROM ACTION MASK
            is_valid_action = False
            exit_counter = 0
            while(is_valid_action == False and exit_counter<=3000):
                action_number = np.random.randint(0, self.num_actions)
                exit_counter +=1
                if action_mask[action_number] == 1:
                    is_valid_action = True

            return action_number, full_predictions[action_number]

        else:
            if isinstance(self.state_space_shape, tuple):
                state = state.reshape((1,) + self.state_space_shape)
            else:
                state = state.reshape(1, self.state_space_shape)

            legal_moves = self._get_legal_moves(full_predictions,action_mask)
            print(max(legal_moves))
            action_to_select = np.argmax(legal_moves)
            return action_to_select, full_predictions[action_to_select]


    def train(self):
        # Convert memory to numpy arrays

        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        #samples = self.memory.buffer

        states = np.array([s[0] for s in minibatch])
        actions = np.array([s[1] for s in minibatch])
        rewards = np.array([s[2] for s in minibatch])
        next_states = np.array([s[3] for s in minibatch])
        dones = np.array([s[4] for s in minibatch])
        old_logprobs = np.array([s[5] for s in minibatch])
        
        # Calculate values and advantages
        values = self.critic(tf.convert_to_tensor(states, dtype=tf.float32)).numpy().flatten()
        advantages = self.compute_advantages(rewards, values, dones)
        returns = advantages + values
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_logprobs = tf.convert_to_tensor(old_logprobs, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Train for multiple epochs
        for _ in range(self.epochs):
            
            # Shuffle indices
            indices = tf.range(start=0, limit=tf.shape(states)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            
            # Mini-batch updates
            for start in range(0, tf.shape(states)[0], self.batch_size):
                end = start + self.batch_size
                batch_indices = shuffled_indices[start:end]
                
                batch_states = tf.gather(states, batch_indices)
                batch_actions = tf.gather(actions, batch_indices)
                batch_old_logprobs = tf.gather(old_logprobs, batch_indices)
                batch_advantages = tf.gather(advantages, batch_indices)
                batch_returns = tf.gather(returns, batch_indices)
                
                # Update networks
                self.train_step(batch_states, batch_actions, 
                               batch_old_logprobs, batch_advantages, batch_returns)
    

    def save_full_model(self, episode, save_critic = True):

        self.actor.save(f'ppo_actor_{episode}', save_format = "tf")

        if save_critic:
            self.critic.save(f'ppo_critic_{episode}', save_format = "tf")

    def load_full_models(self, path_to_actor, critic_path = ''):

        self.actor = load_model(path_to_actor)

        if critic_path != '':
            self.critic = load_model(critic_path)


    def train_step(self, states, actions, old_logprobs, advantages, returns):

        # Update critic (unchanged)
        with tf.GradientTape() as tape:
            values = self.critic(states)
            critic_loss = tf.reduce_mean((returns - tf.squeeze(values))**2)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update actor
        with tf.GradientTape() as tape:
            # Get logits from actor
            logits = self.actor(states)
            
            # Create categorical distribution
            dist = tfd.Categorical(logits=logits)
            
            # Calculate new log probabilities
            new_logprobs = dist.log_prob(actions)
            
            # Calculate ratio (importance sampling)
            ratio = tf.exp(new_logprobs - old_logprobs[0])
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Entropy bonus (encourages exploration)
            entropy = tf.reduce_mean(dist.entropy())
            total_loss = actor_loss - self.entropy_coef * entropy

        # Apply gradients
        actor_grads = tape.gradient(total_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return actor_loss, critic_loss, entropy