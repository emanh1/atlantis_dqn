import optuna
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000, deterministic=True, verbose=0):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            continue_training = super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return continue_training

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
    eval_env = VecTransposeImage(eval_env)
    
    hyperparams = optimize_agent(trial)
    
    model = DQN("CnnPolicy", train_env, verbose=0, tensorboard_log="./atlantis_tensorboard/", **hyperparams)
    
    eval_callback = TrialEvalCallback(eval_env, trial, n_eval_episodes=5, eval_freq=15_000)
    
    model.learn(total_timesteps=150_000, tb_log_name="DQN_HPO", callback=eval_callback)
    
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
        
    return eval_callback.last_mean_reward

if __name__ == "__main__":
    print("Starting Optuna hyperparameter optimization...")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        study_name="atlantis_dqn_optimization",
        storage="sqlite:///atlantis_optuna.db",
        load_if_exists=True,
        direction="maximize",
        pruner=pruner
    )
    study.optimize(objective, n_trials=50)
    
    print("Optimization finished.")
    
    print("\nBest 5 trials:")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    
    for i, trial in enumerate(completed_trials[:5]):
        print(f"Rank {i+1}: Trial {trial.number} with Mean Reward: {trial.value}")
        print(f"    Params: {trial.params}")
