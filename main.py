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
from utils import play_training_tournament
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

# TODO: DOPRAJ GO TUURNAMENT LOOP ZA TRAIN MODEL

# TODO: update reward system

# TODO: fix weight update on train function call

env = chess_v6.env(render_mode="human")
env.reset(seed=42)

number_of_actions = env.action_space('player_1').n
observation_space_size = env.observation_space('player_1')['observation'].shape[2]

cnn_model = CNN(number_of_actions,128,1)
cnn_model.compile(Adam(0.01),loss=MeanSquaredError())

player_model = DQN((8,8,111),number_of_actions,cnn_model,cnn_model)

opp_1 = DDQN((8,8,111),number_of_actions,cnn_model,cnn_model)
opp_2 = DQN((8,8,111),number_of_actions,cnn_model,cnn_model)

opp_3 = DDPG((8,8,111),(1,),cnn_model, cnn_model)

opponents = [opp_3,opp_2,opp_1]

wins = dict()
matches_played = 0

#trained_model, wins, round_counter, match_couner = play_training_tournament(player_model,opponents,env,2,1)

def check_dumbass(state) -> bool:

    for row in range(8):
        print('==================')
        for col in range(8):
            
            for i in range(111):

                if i>6 and i<19 and state[row][col][i] == True:
                    
                    print(f'True at: row = {row}, col = {col}, index = {i}')


def count_pieces(state, encodings_by_value : dict):

    piece_count = 0
    pieces_by_type = dict()
    print('=== COUNTING PIECES ===')

    for row in range(8):

        for col in range(8):

            for i in range(111):
                
                 if i in encodings_by_value and state[row][col][i] == True:
                        
                    #print(f'Piece at: row = {row}, col = {col}, index = {i}')
                    piece_count += 1

                    piece_name = encodings_by_value[i]

                    if piece_name not in pieces_by_type:
                        pieces_by_type[piece_name] = 1
                    else:
                        pieces_by_type[piece_name] +=1

    return piece_count,pieces_by_type


def calculate_reward(previous_piece_nums : dict, piece_nums_current : dict, rewards_by_piece: dict) -> int:
    
    for key in previous_piece_nums.keys():
        value = previous_piece_nums[key]
        if piece_nums_current[key] != value:
            reward = rewards_by_piece[key]
            return reward
    
    return -1


for opponent in opponents:

    for match in range(5):

        env.reset()
        previous_number_of_pieces = 32

        # from piece_encodings.py
        pieces_by_type_previous = piece_nums
        initial_state = True


        # OBRATEN AGENT DAVA ZA WIN
        for agent in env.agent_iter():

            piece_taken_in_move = False
            observation, reward, termination, truncation, info = env.last()
            state = observation['observation']
            
            # if termination or truncation:
                
            #     if agent not in wins:
            #         wins[agent] = 1
            #     else:
            #         wins[agent] +=1

            #     print(f'WINNER: {agent}')
            #     break
            # else:

            moves = observation["action_mask"]             

            expanded_state = np.expand_dims(state, axis = 0)
            converted_state = np.array(expanded_state, dtype=float)

            if agent == 'player_1':
                if isinstance(opponent, DDPG):
                        action = opponent.get_action(converted_state,0.01, discrete=True, legal_moves=moves)
                else:
                        action = opponent.get_action(converted_state,0.01,moves)
            else:
                    action = player_model.get_action(converted_state,0.01,moves)

            #TAKE ACTION
            env.step(action)

            new_observation, reward, termination, truncation, info = env.last()

            new_state = new_observation['observation']

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
                
                print('REWARD IS {reward}')
                # update model memory after every move
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

        if match % 2 == 0:
            print('Training model and updating weights')
            player_model.train()

print(wins)
print(matches_played)
env.close()