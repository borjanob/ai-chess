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
from utils.utils import play_training_tournament, play_vs_random, calculate_reward, count_pieces,add_to_logs,play_training_tournament_with_2_agents
from utils.piece_encodings_full import *
import tensorflow as tf
from utils.q_learning import DDPG
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = chess_v6.env(render_mode="human")
env.reset(seed=42)

number_of_actions = env.action_space('player_1').n
observation_space_size = env.observation_space('player_1')['observation'].shape[2]

model_1_model = Agent(number_of_actions,128)
model_1_model.compile(Adam(0.01),loss=MeanSquaredError())

model_1_target = Agent(number_of_actions,128)

p_model = Agent(number_of_actions,128)
p_model.compile(Adam(0.01),loss=MeanSquaredError())

layers = p_model.layers

dueling = DuelingDQN((8,8,111),number_of_actions,layers,batch_size=32)

dqn = DQN((8,8,111),number_of_actions,model_1_model,model_1_target,batch_size=32)

model_2_model = Agent(number_of_actions,128)
model_2_model.compile(Adam(0.01),loss=MeanSquaredError())

model_2_target = Agent(number_of_actions,128)

ddqn = DDQN((8,8,111),number_of_actions,model_2_model,model_2_target)


wins = dict()
matches_played = 0
illegal_moves = 0
avg_rewards = []
moves_timeout = 0
models = [dueling,ddqn,dqn]

models,data = play_training_tournament_with_2_agents(models,env,1,1)


for opponent in models:
    print('================')
    print(f'Playing against {opponent.__class__.__name__}')
    print('================')
    
    for match in range(3):

        env.reset()
        previous_number_of_pieces = 32

        # from piece_encodings.py
        pieces_by_type_previous = piece_nums
        initial_state = True
        move_counter = 0
        for agent in env.agent_iter():
            
            print(f'{agent} making a move')
            piece_taken_in_move = False
            observation, reward, termination, truncation, info = env.last()
            state = observation['observation']
            
            moves = observation["action_mask"]             

            if max(moves) == 0:
                print('No legal moves left')
                break

            expanded_state = np.expand_dims(state, axis = 0)
            converted_state = np.array(expanded_state, dtype=float)

            if agent == 'player_1':
                if isinstance(opponent, DDPG):
                        action = opponent.get_action(converted_state,0.01, discrete=True, legal_moves=moves)
                else:
                        action = opponent.get_action(converted_state,0.01,moves)
            else:
                if isinstance(player_model, DDPG):
                        action = player_model.get_action(converted_state,0.01, discrete=True, legal_moves=moves)
                else:
                        action = player_model.get_action(converted_state,0.01,moves)

            env.step(action)
            move_counter +=1
            new_observation, reward, termination, truncation, info = env.last()

            new_state = new_observation['observation']
            

            # illegal move made
            if moves[action] == 0:

                illegal_moves +=1
                print(f'{agent} made illegal move, terminating game')

                if agent == 'player_0':

                    # give negative reward to model being trained for illegal moves
                    reward = -50
                    player_model.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)

                break


            number_of_pieces_on_board, pieces_by_type = count_pieces(new_state,piece_encodings_by_number)

            if number_of_pieces_on_board != previous_number_of_pieces and initial_state == False:
                piece_taken_in_move = True
            

            if initial_state:
                initial_state = False

            if agent == 'player_0':
                
                if piece_taken_in_move and termination == False:
                    reward = calculate_reward(pieces_by_type_previous,pieces_by_type, rewards_by_piece)

                if termination:
                    reward = 100
                # if reward > 0:
                #     print(f'REWARD IS {reward}')

                # update model memory after every move
                #player_model.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)
            else:

                if piece_taken_in_move and termination == False:
                    # model being trained loses a piece
                    reward = 0 - calculate_reward(pieces_by_type_previous,pieces_by_type, rewards_by_piece)

                if termination:

                    # model being trained loses
                    reward = -100

            if reward != 0:
                print(f'REWARD IS {reward}')

            # update model memory after every move
            player_model.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)



            # if agent == 'player_1' and termination:
            #     reward = -100
            #     # give negative reward on loss
            #     player_model.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)
            
            
            if termination or truncation:
                
                if agent not in wins:
                    wins[agent] = 1
                else:
                    wins[agent] +=1

                print(f'WINNER: {agent}')
                print(f'Number of moves in game = {move_counter}')
                break
            
            # set previous board state to equal current state after change
            if piece_taken_in_move:
                previous_number_of_pieces = number_of_pieces_on_board
                pieces_by_type_previous = pieces_by_type
            
        print('match finished')
        matches_played+=1
        player_model.train()

        if match % 2 == 0:
            print('Updating target model weights')
            player_model.update_target_model()

print(wins)
print(matches_played)
print(illegal_moves)
# wins = play_vs_random(env,player_model,10)
# print(wins)
env.close()