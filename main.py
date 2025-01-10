from pettingzoo.classic import chess_v6
from algorithms.ddqn import DDQN
from algorithms.dqn import DQN
from algorithms.dueling_dqn import DuelingDQN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MSE
from tensorflow import reduce_mean, convert_to_tensor, squeeze, float32, GradientTape
import numpy as np
from model.agent import Agent
from pettingzoo import AECEnv
from utils.utils import play_training_tournament, play_vs_random, calculate_reward, count_pieces,add_to_logs
from utils.piece_encodings_full import *
import tensorflow as tf
import time

# List all physical devices
physical_devices = tf.config.list_physical_devices()
print("Physical devices:", physical_devices)

# List available logical devices
logical_devices = tf.config.list_logical_devices()
print("Logical devices:", logical_devices)

# Check the default device being used
print("TensorFlow is using:", tf.test.gpu_device_name() if tf.test.is_gpu_available() else "CPU")

env = chess_v6.env()
env.reset(seed=42)

number_of_actions = env.action_space('player_1').n
observation_space_size = env.observation_space('player_1')['observation'].shape[2]

dueling_dqn_model = Agent(number_of_actions,128)
dueling_dqn_model.compile(Adam(0.01),loss=MeanSquaredError())

layers = dueling_dqn_model.layers

dueling = DuelingDQN((8,8,111),number_of_actions,layers,batch_size=32)


dqn_model = Agent(number_of_actions,128)
dqn_model.compile(Adam(0.01),loss=MeanSquaredError())

dqn_target = Agent(number_of_actions,128)

dqn = DQN((8,8,111),number_of_actions,dqn_model,dqn_target,batch_size=32)

ddqn_model = Agent(number_of_actions,128)
ddqn_model.compile(Adam(0.01),loss=MeanSquaredError())

ddqn_target = Agent(number_of_actions,128)

ddqn = DDQN((8,8,111),number_of_actions,ddqn_model,ddqn_target)

wins = dict()
matches_played = 0
illegal_moves = 0
avg_rewards = []

dueling.load('duelingdqn_agent_20.weights.h5')

models = [dqn,ddqn,dueling]
#start_time = time.time()
new_models,data = play_training_tournament(models,env,rounds_in_tournament=20,matches_per_opponent=20)
#end_time =  time.time()

#print(f'Time taken: {end_time - start_time}')
# 1.23 mins za 3 modeli po 3 partii
# 0.41 1 model, 3 partii
# 0.13 1 model, 1 partija

env.close()