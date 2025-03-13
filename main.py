from pettingzoo.classic import chess_v6
from algorithms.ddqn import DDQN
from algorithms.dqn import DQN
from algorithms.dueling_dqn import DuelingDQN
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanSquaredError
from model.agent import Agent
from utils.utils import play_matches, play_training_tournament, play_training_tournament_with_2_agents
from utils.piece_encodings_full import *
from algorithms.ppo import PPO
import time
import tensorflow as tf
from tensorflow import float32

env = chess_v6.env()
env.reset(seed=42)

number_of_actions = env.action_space('player_1').n
observation_space_size = env.observation_space('player_1')['observation'].shape[2]

dueling_dqn_model = Agent(number_of_actions,160)
dueling_dqn_model.compile(SGD(0.00001,clipnorm=2),loss=MeanSquaredError())

layers = dueling_dqn_model.layers

dueling = DuelingDQN((8,8,111),number_of_actions,layers,batch_size=64,discount_factor=0.9)


dqn_model = Agent(number_of_actions,384)
dqn_model.compile(SGD(0.000001,clipnorm=2),loss=MeanSquaredError())

dqn_target = Agent(number_of_actions,384)

dqn = DQN((8,8,111),number_of_actions,dqn_model,dqn_target,batch_size=64, discount_factor=0.88)



ddqn_model = Agent(number_of_actions,320)
ddqn_model.compile(Adam(0.000005,clipnorm=2),loss=MeanSquaredError())
ddqn_target = Agent(number_of_actions,320)

ddqn = DDQN((8,8,111),number_of_actions,ddqn_model,ddqn_target, batch_size=56,discount_factor=0.88)

ppo = PPO((8,8,111),number_of_actions,actor_lr= 0.00007, critic_lr=0.00005,epochs=8,num_of_hidden_units_actor=224,
          num_of_hidden_units_critic=352,discount_factor=0.87,batch_size=24)


wins = dict()
matches_played = 0
illegal_moves = 0
avg_rewards = []
dqn.save_full_model(1)
ddqn.save_full_model(1)
dueling.save_full_model(1)
ppo.save_full_model(1)

models = [dqn,ddqn,dueling,ppo]

new_models, _ = play_training_tournament(models,env,matches_per_opponent=16,rounds_in_tournament=20,episodes_for_target_update=2,save_models_time=5)
trained,_ = play_training_tournament_with_2_agents(new_models, env,matches_per_opponent=16,rounds_in_tournament=20,episodes_for_target_update=2,save_models_interval=5)


for i in range(len(trained)):
    for j in range(i+1,len(trained)):

        first_agent= trained[i]
        second_agent = trained[j]

        players = [first_agent,second_agent]

        white_wins, black_wins = play_matches(env,players)


env.close()