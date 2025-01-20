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
from utils.utils import play_training_tournament, play_vs_random, calculate_reward, count_pieces,add_to_logs, play_training_tournament_with_2_agents, play_matches
from utils.piece_encodings_full import *
import tensorflow as tf
import time
import h5py

env = chess_v6.env()
env.reset(seed=42)

number_of_actions = env.action_space('player_1').n
observation_space_size = env.observation_space('player_1')['observation'].shape[2]

dueling_dqn_model = Agent(number_of_actions,128)
dueling_dqn_model.compile(Adam(0.0001,clipnorm=2),loss=MeanSquaredError())

layers = dueling_dqn_model.layers

dueling = DuelingDQN((8,8,111),number_of_actions,layers,batch_size=32)

dqn_model = Agent(number_of_actions,128)
dqn_model.compile(Adam(0.0001,clipnorm=2),loss=MeanSquaredError())

dqn_target = Agent(number_of_actions,128)

dqn = DQN((8,8,111),number_of_actions,dqn_model,dqn_target,batch_size=32)

ddqn_model = Agent(number_of_actions,128)
ddqn_model.compile(Adam(0.0001,clipnorm=2),loss=MeanSquaredError())


ddqn_target = Agent(number_of_actions,128)

ddqn = DDQN((8,8,111),number_of_actions,ddqn_model,ddqn_target)

wins = dict()
matches_played = 0
illegal_moves = 0
avg_rewards = []

dqn.load('full_models/dqn_model_38.h5')
dqn.update_target_model()
ddqn.load('full_models/ddqn_model_38.h5')
ddqn.update_target_model()
dueling.load('full_models/duelingdqn_model_38.h5')
dueling.update_target_model()

models = [dqn,ddqn,dueling]

for i in range(len(models)):

    for j in range(i+1,len(models)):

        first_agent= models[i]
        second_agent = models[j]

        players = [first_agent,second_agent]

        white_wins, black_wins = play_matches(env,players)

env.close()