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
                pass
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



def play_training_tournament(model_to_train: Model, opponents: list[Model], env: chess_v6, matches_per_opponent: int = 10,
        rounds_in_tournament: int = 5, add_random_opponent: bool = True    ) -> dict:
    
    wins = dict()
    matches_played = 0
    rounds_played = 0

    for round in range(rounds_in_tournament):
            
        for opponent in opponents:

            for match in range(matches_per_opponent):

                # reset env before every game
                env.reset()

                for agent in env.agent_iter():

                    observation, reward, termination, truncation, info = env.last()
                    state = observation['observation']

                    if termination or truncation:
                        action = None

                        if agent not in wins:
                            wins[agent] = 1
                        else:
                            wins[agent] +=1

                        break
                    else:

                        moves = observation["action_mask"]             
                        # this is where you would insert your policy
                        expanded_state = np.expand_dims(state, axis = 0)
                        converted_state = np.array(expanded_state, dtype=float)

                        if agent == 'player_1':
                            if isinstance(opponent, DDPG):
                                action = opponent.get_action(converted_state,0.01, discrete=True, legal_moves=moves)
                            else:
                                action = opponent.get_action(converted_state,0.01,moves)
                        else:
                            
                            # predictions = model.predict(test)
                            action = model_to_train.get_action(converted_state,0.01,moves)
                            
                    env.step(action)

                    if agent == 'player_0':
                        new_observation, reward, termination, truncation, info = env.last()
                        new_state = new_observation['observation']

                        # update model memory after every move
                        model_to_train.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)
                    
                    
                        if match % 2 == 0:
                            print('Training model and updating weights')
                            model_to_train.train()
                    


                print('match finished')
                matches_played+=1

        rounds_played += 1

    return model_to_train, wins, rounds_played, matches_played



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
