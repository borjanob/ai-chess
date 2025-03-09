from utils.tuning import find_best_params
import time 


if __name__ == "__main__":

    """
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=42))
    study.optimize(objective, n_trials=10)
    """
    algorithms = ["dqn",'ddqn'] 
    start = time.time()
    best_params = find_best_params(algorithms,50)
    end = time.time()
    print(end-start)
    print(best_params)