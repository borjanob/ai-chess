from pettingzoo.classic import chess_v6
from pettingzoo.utils.conversions import aec_to_parallel
from model.agent import Agent
from algorithms.dqn import DQN
from algorithms.ddqn import DDQN
from algorithms.dueling_dqn import DuelingDQN
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError 
import optuna
from utils.utils import play_vs_random

from sklearn.model_selection import cross_val_score



def objective_dqn(trial):

    learning_rate = trial.suggest_float('lr', 0.000001,0.00001)
    batch_size = trial.suggest_int('batch_size', 16,64, 8)
    discount_factor = trial.suggest_float('discount_factor', 0.85,0.99)
    hidden_units = trial.suggest_int('hidden_units', 128,512, 32)
    optimizers = trial.suggest_categorical('optimizer',[Adam, SGD])
    

    env = chess_v6.env()
    env.reset(seed=42)


    number_of_actions = 4672

    dqn_model = Agent(number_of_actions,number_of_hidden_units=hidden_units)
    dqn_model.compile(optimizers(learning_rate,clipnorm=2),loss=MeanSquaredError())

    dqn_target = Agent(number_of_actions,number_of_hidden_units=hidden_units)

    dqn = DQN((8,8,111),number_of_actions,dqn_model,dqn_target,batch_size=batch_size,discount_factor=discount_factor)

    score = play_vs_random(env,dqn,1)

    return score




def objective_ddqn(trial):

    learning_rate = trial.suggest_float('lr', 0.000001,0.00001)
    batch_size = trial.suggest_int('batch_size', 16,64, 8)
    discount_factor = trial.suggest_float('discount_factor', 0.85,0.99)
    hidden_units = trial.suggest_int('hidden_units', 128,512, 32)
    optimizers = trial.suggest_categorical('optimizer',[Adam, SGD])
    

    env = chess_v6.env()
    env.reset(seed=42)


    number_of_actions = 4672

    ddqn_model = Agent(number_of_actions,number_of_hidden_units=hidden_units)
    ddqn_model.compile(optimizers(learning_rate,clipnorm=2),loss=MeanSquaredError())

    ddqn_target = Agent(number_of_actions,number_of_hidden_units=hidden_units)

    ddqn = DDQN((8,8,111),number_of_actions,ddqn_model,ddqn_target,batch_size=batch_size,discount_factor=discount_factor)

    score = play_vs_random(env,ddqn,1)

    return score


# TODO: FIX OBJECTIVE FUNC FOR DUELING DQN
def objective_dueling(trial):

    learning_rate = trial.suggest_float('lr', 0.000001,0.00001)
    batch_size = trial.suggest_int('batch_size', 16,64, 8)
    discount_factor = trial.suggest_float('discount_factor', 0.85,0.99)
    hidden_units = trial.suggest_int('hidden_units', 128,512, 32)
    optimizers = trial.suggest_categorical('optimizer',[Adam, SGD])
    

    env = chess_v6.env()
    env.reset(seed=42)


    number_of_actions = 4672

    dueling_dqn_model = Agent(number_of_actions,number_of_hidden_units=hidden_units)
    dueling_dqn_model.compile(optimizers(learning_rate,clipnorm=2),loss=MeanSquaredError())

    layers = dueling_dqn_model.layers

    dueling = DuelingDQN((8,8,111),number_of_actions,layers,batch_size=batch_size,discount_factor=discount_factor)


    score = play_vs_random(env,dueling,1)

    return score




def get_best_params(algorithms, number_of_trials) -> dict:

    """

    Function that uses optuna library to find the best hyperparameters
    to use with the DQN, DDQN, Dueling DQN and PPO algorithms for use with chess

    """

    best_params = dict()

    for algorithm in algorithms:

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=42))

        if algorithm == 'dqn':
                study.optimize(objective_dqn, n_trials=number_of_trials)
        elif algorithm == 'ddqn':
                study.optimize(objective_ddqn, n_trials=number_of_trials)
        elif algorithm == 'dueling':
                study.optimize(objective_dueling, n_trials=number_of_trials)
        #elif algorithm =='ppo':
        #        study.optimize(objective_ppo, n_trials=number_of_trials)
        
        #study.optimize(objective, n_trials=10)

        best_params[algorithm] = study.best_params
    
    return best_params







if __name__ == "__main__":

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=42))
    study.optimize(objective, n_trials=10)

    print(study.best_params)