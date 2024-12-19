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
from piece_encodings import * 

def play_vs_random(env, model: Model, number_of_games: int) -> dict:
    env.reset()
    wins = dict()

    for match in range(number_of_games):    
        for agent in env.agent_iter():

            observation, reward, termination, truncation, info = env.last()
            state = observation['observation']

            moves = observation["action_mask"]             

            if max(moves) == 0:
                print('No legal moves left')
                break

            expanded_state = np.expand_dims(state, axis = 0)
            converted_state = np.array(expanded_state, dtype=float)

            if agent == 'player_1':
                action = env.action_space(agent).sample(moves)
            else:
                action = model.get_action(converted_state,0.01,moves)

            #TAKE ACTION
            env.step(action)

            new_observation, reward, termination, truncation, info = env.last()    
            
            if termination or truncation:
                
                if agent not in wins:
                    wins[agent] = 1
                else:
                    wins[agent] +=1

                print(f'WINNER: {agent}')
                break
    return wins


def play_training_tournament(models: list[Model], env: chess_v6, matches_per_opponent: int = 10,
        rounds_in_tournament: int = 5, add_random_opponents: bool = True ):
    
    for round in range(rounds_in_tournament):
        updated_models = []
        print('===============')
        print(f'Round {round}: ')

        for model_to_train in models:
            
            print('===============')
            print(f'{model_to_train.__class__.__name__} being trained')
            print('===============')

            opponents = [ x for x in models if x != model_to_train]

            trained_model, wins_data = _play_tournament_round(model_to_train, opponents, env,matches_per_opponent,add_random_opponents)

            updated_models.append(trained_model)
            
 
            print(f'Stats after training {wins_data}')

            
        models = updated_models

    return models




def _play_tournament_round(model_to_train: Model, opponents: list[Model], env: chess_v6, matches_per_opponent: int = 10,
                add_random_opponent: bool = True    ) -> dict:
    
    wins = dict()

    if add_random_opponent:
        opponents.append('random')

    
    for opponent in opponents:
        print('================')
        print(f'Playing against {opponent.__class__.__name__}')
        print('================')
        
        for match in range(matches_per_opponent):

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
                    
                    if opponent == 'random':
                        action = env.action_space(agent).sample(moves)
                    else:
                            action = opponent.get_action(converted_state,0.01,moves)

                else:
                    action = model_to_train.get_action(converted_state,0.01,moves)


                env.step(action)

                new_observation, reward, termination, truncation, info = env.last()

                new_state = new_observation['observation']
                

                # illegal move made
                if moves[action] == 0:
                    
                    if agent == 'player_0':

                        # give negative reward to model being trained for illegal moves
                        reward = -50
                        model_to_train.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)
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
                    model_to_train.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)
                
                if agent == 'player_1' and termination:
                    reward = -100
                    # give negative reward on loss
                    model_to_train.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)
                
                
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
            model_to_train.train()
            if match % 5 == 0:
                print('Training model and updating weights')
                model_to_train.update_target_model()
    

    return model_to_train, wins



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

    for row in range(8):

        for col in range(8):

            for i in range(111):
                
                 if i in encodings_by_value:
                    
                    piece_name = encodings_by_value[i]

                    if state[row][col][i] == True:
                        
                        piece_count += 1

                        if piece_name not in pieces_by_type:
                            pieces_by_type[piece_name] = 1
                        else:
                            pieces_by_type[piece_name] +=1
                    else:
                        if piece_name not in pieces_by_type:
                            pieces_by_type[piece_name] = 0 

    
    # for name in encodings_by_name:
    #     if name not in pieces_by_type[name]:
    #         pieces_by_type[name] = 0


    return piece_count,pieces_by_type


def calculate_reward(previous_piece_nums : dict, piece_nums_current : dict, rewards_by_piece: dict) -> int:
    
    for key in previous_piece_nums.keys():
        value = previous_piece_nums[key]
        if piece_nums_current[key] != value:
            reward = rewards_by_piece[key]
            return reward
    
    return -1
