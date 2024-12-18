from pettingzoo.classic import chess_v6
from q_learning import DQN, DDPG, DDQN
from tensorflow.keras.layers import Input, Dense, Concatenate, Conv2D, Flatten
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MSE
from tensorflow import reduce_mean, convert_to_tensor, squeeze, float32, GradientTape
import numpy as np
from cnn_model import CNN
from tensorflow.keras.models import Model 
from pettingzoo import AECEnv
from utils import play_training_tournament, play_vs_random, calculate_reward, count_pieces
from piece_encodings import *

# SEGDE IMAS SMENETO WEIGHTS DA NE SE UPDATE NA POCETOK 
# VO ALGORITMITE OD Q_LEARNING.PY
# SMENA I VO DDPG ZA DA IMA UBAV SHAPE I DODADE FLATTEN

# -----DONE-----
# TODO: DODAJ GI I VO DDPG DA ZEMA AKCIJA SAMO OD DOZVOLENITE 
# KAKO SO IMAS I VO DRUGITE ALGORTIMI SO MNOZENJE NA ACTION MASK 
# -----DONE-----

# -----DONE-----
# TODO: VO ACTOR-CRITIC ALGORITMO VRAKA 8*8*num_of_actions AKCII VO PREDICT
# SREDI GO DA SE NAMALAT NA TOCNIO BROJ AKCII
# -----DONE-----

# -----DONE-----
# TODO: update reward system
# -----DONE-----

# -----DONE-----
# TODO: ZA PLAYER1 AKO NAPRAJ ILLEGAL SAMO DA PRAJ EXIT, A PLAYER 0 AKO NAPRAJ ILLEGAL DA MU DAVA NEGATIVE REWARD
# -----DONE-----

# TODO: smeni za train i update target model

# TODO: FIX TRAIN CALL ON A2C

# TODO: KO KE SE NAPRAJ ILLEGAL MOVE DA POBEDI DRUGIO A NE TOJ SO E NA RED

# -----DONE-----
# TODO: DOPRAJ GO TUURNAMENT LOOP ZA TRAIN MODEL
# -----DONE-----

# vidi za so se koristi target model
# TODO: za konsultacii so da prasas
# odi na kons da vidis za DDPG kako e kodo napisan da ti objasni zosto te ebava
# zosto ima suqeeze vo gradientape 
# dali ti trebat tebe kako touple ili mozis samo da gi pustis inputs deka output shape ti e fixiran vo CNN klasata
# kako mozis da gi pustis site od cel batch naednas vo DDPG
# dali treba razlicna arhitektura na model za sekoj algoritam


env = chess_v6.env(render_mode="human")
env.reset(seed=42)

number_of_actions = env.action_space('player_1').n
observation_space_size = env.observation_space('player_1')['observation'].shape[2]


cnn_model_initial = CNN(number_of_actions,128,1)
cnn_model_initial.compile(Adam(0.01),loss=MeanSquaredError())


cnn_model = CNN(number_of_actions,128,1)
cnn_model.compile(Adam(0.01),loss=MeanSquaredError())

cnn_model_opp1 = CNN(number_of_actions,128,1)
cnn_model_opp1.compile(Adam(0.01),loss=MeanSquaredError())


cnn_model_opp2 = CNN(number_of_actions,128,1)
cnn_model_opp2.compile(Adam(0.01),loss=MeanSquaredError())

cnn_model_opp3 = CNN(number_of_actions,128,1)
cnn_model_opp3.compile(Adam(0.01),loss=MeanSquaredError())

#player_model = DQN((8,8,111),number_of_actions,cnn_model,cnn_model,batch_size=32)

opp_3 = DDQN((8,8,111),number_of_actions,cnn_model_opp1,cnn_model_opp1)
opp_2 = DQN((8,8,111),number_of_actions,cnn_model_opp2,cnn_model_opp2)

player_model = DDPG((8,8,111),(1,),number_of_actions,cnn_model_opp3, cnn_model_opp3)

opponents = [opp_2,opp_3]

wins = dict()
matches_played = 0


avg_rewards = []


models = [opp_2,opp_3,player_model]

#testing = play_training_tournament(models,env,1,1)


for opponent in opponents:
    print('================')
    print(f'Playing against {opponent.__class__.__name__}')
    print('================')
    
    for match in range(3):

        env.reset()
        previous_number_of_pieces = 32

        # from piece_encodings.py
        pieces_by_type_previous = piece_nums
        initial_state = True

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

            new_observation, reward, termination, truncation, info = env.last()

            new_state = new_observation['observation']
            

            # illegal move made
            if moves[action] == 0:
                
                if agent == 'player_0':

                    # give negative reward to model being trained for illegal moves
                    reward = -50
                    player_model.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)
                    #wins['player_1'] += 1
                # else:
                #      wins['player_0'] += 1

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
                if reward > 0:
                    print(f'REWARD IS {reward}')

                # update model memory after every move
                player_model.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)
            
            if agent == 'player_1' and termination:
                reward = -100
                # give negative reward on loss
                player_model.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)
            
            
            if termination or truncation:
                
                if agent not in wins:
                    wins[agent] = 1
                else:
                    wins[agent] +=1

                print(f'WINNER: {agent}')
                break
            
            # set previous board state to equal current state after change
            if piece_taken_in_move:
                previous_number_of_pieces = number_of_pieces_on_board
                pieces_by_type_previous = pieces_by_type
            
        print('match finished')
        matches_played+=1
        player_model.train()

        if match % 1 == 0:
            print('Training model and updating weights')
            player_model.update_target_model()

print(wins)
print(matches_played)

# wins = play_vs_random(env,player_model,10)
# print(wins)
env.close()