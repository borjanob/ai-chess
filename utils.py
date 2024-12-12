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