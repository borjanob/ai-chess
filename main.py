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



env = chess_v6.env(render_mode="human")
env.reset(seed=42)

number_of_actions = env.action_space('player_1').n
observation_space_size = env.observation_space('player_1')['observation'].shape[2]

model_1_model = Agent(number_of_actions,128)
model_1_model.compile(Adam(0.01),loss=MeanSquaredError())

model_1_target = Agent(number_of_actions,128)

p_model = Agent(number_of_actions,128)
p_model.compile(Adam(0.01),loss=MeanSquaredError())

p_target = Agent(number_of_actions,128)


layers = p_model.layers

player_model = DuelingDQN((8,8,111),number_of_actions,layers,batch_size=32)

model_1 = DQN((8,8,111),number_of_actions,model_1_model,model_1_target,batch_size=32)

model_2_model = Agent(number_of_actions,128)
model_2_model.compile(Adam(0.01),loss=MeanSquaredError())

model_2_target = Agent(number_of_actions,128)

model_2 = DDQN((8,8,111),number_of_actions,model_2_model,model_2_target)

wins = dict()
matches_played = 0
illegal_moves = 0
avg_rewards = []

models = [model_1,player_model]

new_models,data = play_training_tournament(models,env,rounds_in_tournament=5,matches_per_opponent=1,add_random_opponents=False)

env.close()