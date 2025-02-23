from pettingzoo.classic import chess_v6
import numpy as np
from model.agent import Agent
from tensorflow.keras.models import Model 
from pettingzoo import AECEnv
from utils.piece_encodings_full import * 
import random
from algorithms.ppo import PPO

def play_vs_random(env, model: Model, number_of_games: int) -> dict:


    """
    Plays an :number_of_games episodes of env against random actions
    """

    env.reset()
    wins = dict()
    rewards = []
    rewards_in_match = 0
    moves_in_match = 0



    for match in range(number_of_games):

        rewards_in_match = 0
        moves_in_match = 0
        
        env.reset()
        previous_number_of_pieces = 32

        pieces_by_type_previous = piece_nums
        initial_state = True

        for agent in env.agent_iter():

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
                action = env.action_space(agent).sample(moves)
            else:

                if isinstance(model, PPO):
                    action, probability = model.get_action(converted_state,0.01,moves)
                else:

                    action = model.get_action(converted_state,0.01,moves)

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
                    
                    moves_in_match+=1

                    if piece_taken_in_move and termination == False:
                        reward = calculate_reward(pieces_by_type_previous,pieces_by_type, rewards_by_piece)

                    if termination:
                        reward = 100
                    if reward > 0:
                        print(f'REWARD IS {reward}')

                    # update model memory after every move
                    if isinstance(model, PPO):
                        model.update_memory(state,action,reward,new_state,1 if termination or truncation else 0, probability)
                    else:
                        model.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)

            rewards_in_match += reward

            if termination or truncation:

                print(f'WINNER: {agent}')
                break

             # set previous board state to equal current state after change
            if piece_taken_in_move:
                previous_number_of_pieces = number_of_pieces_on_board
                pieces_by_type_previous = pieces_by_type
                
        #print('match finished')
        model.train()
        reward_avg = rewards_in_match / moves_in_match
        rewards.append(reward_avg) 
        if (match+1) % 2 == 0 and isinstance(model, PPO) == False:
            print('Updating target model')
            model.update_target_model()

    avg_reward =  np.mean(rewards)

    return avg_reward


def play_matches(env: chess_v6, players : list, number_of_games: int = 10, logs_file_name = 'logs/matches.txt'):

    white_wins = dict()
    black_wins = dict()

    if len(players) != 2:
        raise Exception('2 players are required to play a match')


    first_agent_name = players[0].__class__.__name__
    second_agent_name = players[1].__class__.__name__

    white_player = players[0]
    black_player = players[1]
    switched = False

    game_to_switch = number_of_games/2

    white_player_name = white_player.__class__.__name__
    black_player_name = black_player.__class__.__name__
    matches_ended_in_illegal_moves = 0

    for game in range(number_of_games):

        print('===============')
        print(f'Starting {game} out of {number_of_games}')
        print('===============')

        if game >= game_to_switch and switched == False:

            white_player = players[1]
            black_player = players[0]
            
            white_player_name = white_player.__class__.__name__
            black_player_name = black_player.__class__.__name__

            switched = True

        env.reset()

        print('===============')
        print(f'{white_player_name} playing with white , {black_player_name} playing with black')
        print('===============')



        for agent in env.agent_iter():
                
            print(f'{agent} making a move')

            observation, reward, termination, truncation, info = env.last()
            state = observation['observation']
                
            moves = observation["action_mask"]             

            if max(moves) == 0:
                print('No legal moves left')

                if agent == 'player_1':
                    
                    if white_player_name not in white_wins:
                        white_wins[white_player_name] = 1
                    else:
                        white_wins[white_player_name] +=1

                    print(f'WINNER: {white_player_name}')
                
                else:

                    if black_player_name not in black_wins:
                        black_wins[black_player_name] = 1
                    else:
                        black_wins[black_player_name] +=1

                    print(f'WINNER: {black_player_name}')

                break

            expanded_state = np.expand_dims(state, axis = 0)
            converted_state = np.array(expanded_state, dtype=float)

            if agent == 'player_1':
                action = black_player.get_action(converted_state,0.1,moves)

            else:
                action = white_player.get_action(converted_state,0.1,moves)


            env.step(action)

            new_observation, reward, termination, truncation, info = env.last()

            new_state = new_observation['observation']
                
            # illegal move made
            if moves[action] == 0:
                    
                matches_ended_in_illegal_moves +=1
                print(f'Illegal move made, terminating game!')
                break

                
            if termination or truncation:
                
                if agent == 'player_0':
                    
                    if white_player_name not in white_wins:
                        white_wins[white_player_name] = 1
                    else:
                        white_wins[white_player_name] +=1

                    print(f'WINNER: {white_player_name}')
                
                else:

                    if black_player_name not in black_wins:
                        black_wins[black_player_name] = 1
                    else:
                        black_wins[black_player_name] +=1

                    print(f'WINNER: {black_player_name}')

                break
    
    first_agent_wins_as_white = white_wins[first_agent_name] if first_agent_name in white_wins else 0 
    first_agent_wins_as_black = black_wins[first_agent_name] if first_agent_name in black_wins else 0
    second_agent_wins_as_white = white_wins[second_agent_name] if second_agent_name in white_wins else 0 
    second_agent_wins_as_black = black_wins[second_agent_name] if second_agent_name in black_wins else 0

    round_info = f'{first_agent_name} playing against {second_agent_name} for {number_of_games} games'

    first_agent_info = f'{first_agent_name} won {first_agent_wins_as_white} playing with white, {first_agent_wins_as_black} playing with black'
    second_agent_info = f'{second_agent_name} won {second_agent_wins_as_white} playing with white, {second_agent_wins_as_black} playing with black'

    to_add = [round_info,first_agent_info,second_agent_info]

    add_to_logs(filename=logs_file_name, content=to_add)


    return white_wins, black_wins


def add_to_logs(filename, content):

    """
    Appends the given content to a file, starting on a new line.

    Args:
        filename (str): The name of the logs file
        content (str): The content to be added to logs
    """
    try:
        with open(filename, 'a') as file:
            if isinstance(content, list):
                for line in content:
                    file.write(f"\n{line}")
            else:
                file.write(f"\n{content}")  # Add a newline before the content
        print(f"Logs updated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def play_training_tournament(models: list, env: chess_v6, matches_per_opponent: int = 10,
        rounds_in_tournament: int = 5,episodes_for_target_update:int = 5, add_random_opponents: bool = True, logs_file_name = 'logs/tournament_logs.txt' ):
    

    """
    Defines :models playing against each other in a tournament setting
    """

    rewards_data = dict()
    moves_data = dict()
    data = dict()
    for round in range(rounds_in_tournament):
        updated_models = []
        print('===============')
        print(f'Round {round}: ')

        random.shuffle(models)

        for model_to_train in models:
            
            print('===============')
            print(f'{model_to_train.__class__.__name__} being trained')
            print('===============')

            opponents = [ x for x in models if x != model_to_train]

            trained_model, wins, avg_moves, avg_rewards, total_number_of_games,illegals = _play_tournament_round(model_to_train, opponents, env,matches_per_opponent, episodes_for_target_update, add_random_opponents)

            updated_models.append(trained_model)

            print(f'Stats after round {round}: wins = {wins} with {illegals} illegal move timeouts, average number of moves per win = {avg_moves}, average reward per move = {avg_rewards} ')
            
            info_for_round = f'Model = {model_to_train.__class__.__name__}, Round = {round}, won {wins} games out of {total_number_of_games}, average number of moves per win in this round = {avg_moves}, average reward per move in this round= {avg_rewards} '

            add_to_logs(logs_file_name,info_for_round)

            if model_to_train.__class__.__name__ not in rewards_data:
                rewards_data[model_to_train.__class__.__name__] = avg_rewards
                moves_data[model_to_train.__class__.__name__] = avg_moves
            else:
                rewards_data[model_to_train.__class__.__name__] += avg_rewards
                moves_data[model_to_train.__class__.__name__] += avg_moves

            if (round+1) % 2 == 0:
                model_to_train.save('agent',round + 1)

        models = updated_models
        add_to_logs(logs_file_name,'')


    for model in models:
        avg_rewards_for_model = rewards_data[model_to_train.__class__.__name__] / rounds_in_tournament
        avg_moves_for_model = moves_data[model_to_train.__class__.__name__] / rounds_in_tournament

        data[model.__class__.__name__] = (avg_rewards_for_model,avg_moves_for_model)

    return models, data


def _play_tournament_round(model_to_train: Model, opponents: list, env: chess_v6, matches_per_opponent: int = 10,
            episodes_for_target_update: int = 5, add_random_opponent: bool = True    ) -> dict:
    

    """"
    Plays one round of matches in a tournament setting
    """

    wins_by_player = dict()

    if add_random_opponent:
        opponents.append('random')

    moves_in_matches = 0
    reward_in_matches = 0
    moves_in_wins = 0
    
    matches_ended_in_illegal_moves = 0
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
                    
                    matches_ended_in_illegal_moves +=1
                    print(f'{agent} made illegal move, terminating game')

                    if agent == 'player_0':

                        # give negative reward to model being trained for illegal moves
                        reward = -50
                        model_to_train.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)

                    break
                        


                number_of_pieces_on_board, pieces_by_type = count_pieces(new_state,piece_encodings_by_number)

                if number_of_pieces_on_board != previous_number_of_pieces and initial_state == False:
                    piece_taken_in_move = True
                

                if initial_state:
                    initial_state = False

                if agent == 'player_0':
                    
                    moves_in_matches+=1

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
                
                reward_in_matches += reward
                
                
                if termination or truncation:
                    
                    if agent not in wins_by_player:
                        wins_by_player[agent] = 1
                    else:
                        wins_by_player[agent] +=1

                    print(f'WINNER: {agent}')
                    break
                
                # set previous board state to equal current state after change
                if piece_taken_in_move:
                    previous_number_of_pieces = number_of_pieces_on_board
                    pieces_by_type_previous = pieces_by_type
                
            #print('match finished')
            model_to_train.train()

            if (match+1) % episodes_for_target_update == 0:
                print('Updating target model')
                model_to_train.update_target_model()
    

    games_played = len(opponents) * matches_per_opponent

    avg_moves_in_round = moves_in_matches / games_played
    avg_rewards_per_move = reward_in_matches / moves_in_matches

    wins = wins_by_player['player_0'] if 'player_0' in wins_by_player else 0
    total_number_of_games = len(opponents) * matches_per_opponent

    return model_to_train, wins, avg_moves_in_round, avg_rewards_per_move, total_number_of_games, matches_ended_in_illegal_moves



def play_training_tournament_with_2_agents(models: list, env: chess_v6, matches_per_opponent: int = 10,
        rounds_in_tournament: int = 5,episodes_for_target_update:int = 5, add_random_opponent: bool = True, logs_file_name = 'logs/tournament_logs_2.txt' ):
    

    """
    Defines :models playing against each other in a tournament setting while updating both models
    """

    rewards_data = dict()
    moves_data = dict()
    data = dict()
    for round in range(rounds_in_tournament):

        print('===============')
        print(f'Round {round}: ')

        random.shuffle(models)

        updated_models = _play_tournament_round_update_2_agents(models, env,round,matches_per_opponent, episodes_for_target_update, add_random_opponent, logs_file_name)
       
        models = updated_models

        # if model_to_train.__class__.__name__ not in rewards_data:
        #         rewards_data[model_to_train.__class__.__name__] = avg_rewards
        #         moves_data[model_to_train.__class__.__name__] = avg_moves
        # else:
        #         rewards_data[model_to_train.__class__.__name__] += avg_rewards
        #         moves_data[model_to_train.__class__.__name__] += avg_moves

        if (round + 1) % 5 == 0:
                for model in models:
                    model.save('agent', round + 21)

            

    # for model in models:
    #     avg_rewards_for_model = rewards_data[model_to_train.__class__.__name__] / rounds_in_tournament
    #     avg_moves_for_model = moves_data[model_to_train.__class__.__name__] / rounds_in_tournament

    #     data[model.__class__.__name__] = (avg_rewards_for_model,avg_moves_for_model)

    return models, data


def _play_tournament_round_update_2_agents(models: list, env: chess_v6,round:int, matches_per_opponent: int = 10,
            episodes_for_target_update: int = 5, add_random_opponent: bool = True, logs_file_name = 'logs/tournament_logs_2.txt'    ) -> dict:
    

    """"
    Plays one round of matches in a tournament setting where both white and black players weights are updated
    """

    wins_by_player = dict()

    if add_random_opponent:
        models.append('random')

    moves_in_matches = 0
    reward_in_matches = 0
    moves_in_wins = 0
    updated_opponents = []
    matches_ended_in_illegal_moves = 0

    # every model is the white player against every other model

    for white_player in models:

        if white_player == 'random':
            continue

        print('================')
        print(f'Playing with white is {white_player.__class__.__name__}')
        print('================')
        
        opponents = [ x for x in models if x != white_player]

        for opponent in opponents:
            
            print('================')
            print(f'Playing with  {opponent.__class__.__name__} as black')
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
                        action = white_player.get_action(converted_state,0.01,moves)


                    env.step(action)

                    new_observation, reward, termination, truncation, info = env.last()

                    new_state = new_observation['observation']

                    reward_for_black = reward
                    reward_for_white = reward

                    # illegal move made
                    if moves[action] == 0:
                        
                        matches_ended_in_illegal_moves +=1
                        print(f'{agent} made illegal move, terminating game')

                        if agent == 'player_0':

                            # give negative reward to model being trained for illegal moves
                            reward = -50
                            white_player.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)

                        break
                            


                    number_of_pieces_on_board, pieces_by_type = count_pieces(new_state,piece_encodings_by_number)

                    if number_of_pieces_on_board != previous_number_of_pieces and initial_state == False:
                        piece_taken_in_move = True
                    

                    if initial_state:
                        initial_state = False

                    if agent == 'player_0':
                        
                        moves_in_matches+=1

                        if piece_taken_in_move and termination == False:
                            reward_for_white = calculate_reward(pieces_by_type_previous,pieces_by_type, rewards_by_piece)

                        if termination:
                            reward_for_white = 100
                        if reward_for_white > 0:
                            print(f'REWARD IS {reward_for_white}')
                            reward_for_black = 0 - reward_for_white

                            # give negative reward to opponent on loss or lost piece
                            if opponent != 'random':
                                opponent.update_memory(state,action,reward_for_black,new_state, 1 if termination or truncation else 0)

                        # update model memory after every move
                        white_player.update_memory(state,action,reward_for_white,new_state, 1 if termination or truncation else 0)
                    
                    

                    if agent == 'player_1':

                        if piece_taken_in_move and termination == False:
                            reward_for_black = calculate_reward(pieces_by_type_previous,pieces_by_type, rewards_by_piece)

                        if termination:
                            reward_for_black = 100
                        if reward_for_black > 0:

                            print(f'REWARD IS {reward_for_black}')
                            reward_for_white = 0 - reward_for_black
                            
                            # give negative reward to opponent on loss or lost piece
                            white_player.update_memory(state,action,reward_for_white,new_state, 1 if termination or truncation else 0)

                        # update model memory after every move

                        if opponent != 'random':

                            opponent.update_memory(state,action,reward_for_black,new_state, 1 if termination or truncation else 0)

                    reward_in_matches += reward_for_white if reward_for_white > reward_for_black else reward_for_black
                    
                    
                    if termination or truncation:
                        
                        if agent not in wins_by_player:
                            wins_by_player[agent] = 1
                        else:
                            wins_by_player[agent] +=1

                        print(f'WINNER: {agent}')
                        break
                    
                    # set previous board state to equal current state after change
                    if piece_taken_in_move:
                        previous_number_of_pieces = number_of_pieces_on_board
                        pieces_by_type_previous = pieces_by_type
                    
                #print('match finished')
                white_player.train()

                if opponent != 'random':
                    opponent.train()

                if match % episodes_for_target_update == 0:
                    print('Updating both agents')
                    white_player.update_target_model()

                    if opponent != 'random':
                        opponent.update_target_model()  
        
        
        games_played = len(opponents) * matches_per_opponent

        avg_moves_in_round = moves_in_matches / games_played
        avg_rewards_per_move = reward_in_matches / moves_in_matches

        wins = wins_by_player['player_0'] if 'player_0' in wins_by_player else 0
        total_number_of_games = len(opponents) * matches_per_opponent

        print(f'Stats after round {round + 1}: wins = {wins} with {matches_ended_in_illegal_moves} illegal move timeouts, average number of moves per win = {avg_moves_in_round}, average reward per move = {avg_rewards_per_move} ')

        info_for_round = f'Model = {white_player.__class__.__name__}, Round = {round + 1}, won {wins} games out of {total_number_of_games}, average number of moves per win in this round = {avg_moves_in_round}, average reward per move in this round= {avg_rewards_per_move} '

        add_to_logs(logs_file_name,info_for_round)  

        #models = [x for x in models if x != 'random']
        if add_random_opponent:
            models.remove('random')

    return models


def check_dumbass(state) -> bool:

    for row in range(8):
        print('==================')
        for col in range(8):
            
            for i in range(111):

                if i>6 and i<19 and state[row][col][i] == True:
                    
                    print(f'True at: row = {row}, col = {col}, index = {i}')


def count_pieces(state, encodings_by_value : dict):


    """
    Count number of pieces  and piece types on the board
    """

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

    """
    Calculate reward to give to agent on action
    """
    
    for key in previous_piece_nums.keys():
        value = previous_piece_nums[key]
        if piece_nums_current[key] != value:
            reward = rewards_by_piece[key]
            return reward
    
    return -1
