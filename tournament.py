
wins = dict()
matches_played = 0

for opponent in opponents:

    for match in range(1):
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
                    action = player_model.get_action(converted_state,0.01,moves)
                    
            env.step(action)

            if agent == 'player_0':
                new_observation, reward, termination, truncation, info = env.last()
                new_state = new_observation['observation']

                # update model memory after every move
                player_model.update_memory(state,action,reward,new_state, 1 if termination or truncation else 0)
            
             
                if match % 2 == 0:
                    print('Training model and updating weights')
                    player_model.train()
            


        print('match finished')
        matches_played+=1
