import optuna
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

def optimize_agent(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'buffer_size': trial.suggest_categorical('buffer_size', [10_000, 50_000, 100_000]),
        "learning_starts": trial.suggest_categorical("learning_starts", [1_000, 5_000, 10_000]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
        'exploration_fraction': trial.suggest_float('exploration_fraction', 0.05, 0.2),
        'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.05),
    }

def objective(trial):
    env_id = "ALE/Atlantis-v5"
    
    train_env = make_atari_env(env_id, n_envs=1, seed=0)
    train_env = VecFrameStack(train_env, n_stack=4)
    
    eval_env = make_atari_env(env_id, n_envs=1, seed=0)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    hyperparams = optimize_agent(trial)
    
    model = DQN("CnnPolicy", train_env, verbose=0, tensorboard_log="./atlantis_tensorboard/", **hyperparams)
    
    model.learn(total_timesteps=1_000_000, tb_log_name="DQN_HPO")
    
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
    
    return mean_reward

if __name__ == "__main__":
    print("Starting Optuna hyperparameter optimization...")
    study = optuna.create_study(
        study_name="atlantis_dqn_optimization",
        storage="sqlite:///atlantis_optuna.db",
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective, n_trials=50)
    
    print("Optimization finished.")
    print("Best hyperparameters:", study.best_params)
    print("Best mean reward:", study.best_value)
