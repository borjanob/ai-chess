from pettingzoo.classic import chess_v6
from model.agent import Agent
from algorithms.dqn import DQN
from algorithms.ddqn import DDQN
from algorithms.dueling_dqn import DuelingDQN
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import MeanSquaredError
import optuna
from utils.utils import play_vs_random
from algorithms.ppo import PPO
from utils.utils import add_to_logs
import matplotlib.pyplot as plt

"""
def get_optimizable_params(algorithm) -> []:

    if algorithm == 'dqn':
        return ['lr','batch_size','discount_factor']
    if algorithm == 'ddqn':
                study.optimize(objective_ddqn, n_trials=number_of_trials)
    if algorithm == 'dueling':
                study.optimize(objective_dueling, n_trials=number_of_trials)
    if algorithm =='ppo':
                study.optimize(objective_ppo, n_trials=number_of_trials)
"""
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



def objective_ppo(trial):

    learning_rate_actor = trial.suggest_float('lr_actor', 0.000001,0.0001)
    learning_rate_critic = trial.suggest_float('lr_critic', 0.000001,0.0001)
    batch_size = trial.suggest_int('batch_size', 16,64, 8)
    discount_factor = trial.suggest_float('discount_factor', 0.85,0.99)
    hidden_units_actor = trial.suggest_int('hidden_units_actor', 128,512, 32)
    hidden_units_critic = trial.suggest_int('hidden_units_critic', 128,512, 32)
    epochs = trial.suggest_int('epochs', 3,10)
    #optimizers = trial.suggest_categorical('optimizer',[Adam, SGD])
    

    env = chess_v6.env()
    env.reset(seed=42)

    number_of_actions = 4672
    
    ppo = PPO((8,8,111), number_of_actions,discount_factor=discount_factor, actor_lr= learning_rate_actor, critic_lr= learning_rate_critic,
              batch_size=batch_size, num_of_hidden_units_actor = hidden_units_actor, num_of_hidden_units_critic = hidden_units_critic,  
              epochs=epochs)
    
    score = play_vs_random(env,ppo,1)

    return score

def visualize_tuning(study: optuna.study,algorithm, params: list = None) -> None:

    if algorithm == 'ppo':  
        fig = optuna.visualization.plot_timeline(study)
        fig.show()
        #fig.savefig('test_fig.png')
        
    fig = optuna.visualization.plot_edf(study)
    fig.show()
    plt = optuna.visualization.plot_optimization_history(study)
    plt.show()

    plt = optuna.visualization.plot_param_importances(study)
    plt.show()
    if params:
        fig = optuna.visualization.plot_rank(study,params = params)
        fig.show()
        if algorithm == 'ppo':
            plt = optuna.visualization.plot_slice(study, params = params)
            plt.show()
    
def find_best_params(algorithms, number_of_trials, visualizations = None) -> dict:

    """
    Function that uses optuna library to find the best hyperparameters for passed models
    Params:
    algorithms: [] - algorithms that need hyperparameters found
    number_of_tirals: int - number of trials per opmiziation
    games_per_trial: int - number of games played in each optimization trial

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
        elif algorithm =='ppo':
            study.optimize(objective_ppo, n_trials=number_of_trials)
               
        params = list(study.best_params.keys())
        
        visualize_tuning(study,algorithm,params = params)
        best_params[algorithm] = study.best_params
    
    add_to_logs(f"logs/best_hyperparameters_{number_of_trials}_number_of_trials.txt", best_params)

    return best_params



