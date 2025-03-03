from pettingzoo.classic import chess_v6
from algorithms.ddqn import DDQN
from algorithms.dqn import DQN
from algorithms.dueling_dqn import DuelingDQN
from tensorflow.keras.optimizers import Adam
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

ppo = PPO((8,8,111),number_of_actions)


wins = dict()
matches_played = 0
illegal_moves = 0
avg_rewards = []

zeros = tf.zeros((1,8,8,111), dtype=float32)

models = [dqn,ddqn,dueling,ppo]
start = time.time()
#new_models, _ = play_training_tournament(models,env,1,1,1,save_models_time=1)
print('STOP')

dqn.model.predict(zeros)
ddqn.model.predict(zeros)
ppo.actor.predict(zeros)
ppo.critic.predict(zeros)
dueling.model.predict(zeros)
"""
dqn.save_full_model(1)
ddqn.save_full_model(1)
dueling.save_full_model(1)
ppo.save_full_model(1)
"""
dqn.load_full_model('dqn_model_1')
ddqn.load_full_model('ddqn_model_1')
dueling.load_full_model('duelingdqn_model_1')
ppo.load_full_model('ppo_actor_1')

#new_models, _ = play_training_tournament(models,env,2,1,2)
end = time.time()
"""
for i in range(len(models)):

    for j in range(i+1,len(models)):

        first_agent= models[i]
        second_agent = models[j]

        players = [first_agent,second_agent]

        white_wins, black_wins = play_matches(env,players)
"""
"""
94 sec - 16 games
5.9s - 1 game


"""
print( end- start)
env.close()