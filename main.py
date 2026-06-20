import os
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    print("Setting up the Atlantis environment...")
    
    env_id = "ALE/Atlantis-v5"
    num_envs = 1
    
    env = make_atari_env(env_id, n_envs=num_envs, seed=0)
    
    # Frame-stacking with 4 frames to capture velocity
    env = VecFrameStack(env, n_stack=4)
    
    model_path = "atlantis_dqn_model"
    model_file = f"{model_path}.zip"
    
    if os.path.exists(model_file):
        print(f"Found existing model at {model_file}. Resuming training...")
        model = DQN.load(model_file, env=env, tensorboard_log="./atlantis_tensorboard/")
    else:
        print("Initializing new DQN agent...")
        model = DQN(
            "CnnPolicy",
            env,
            verbose=1,
            buffer_size=100_000,
            learning_starts=10_000,
            batch_size=32,
            learning_rate=1e-4,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            tensorboard_log="./atlantis_tensorboard/",
        )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/",
        name_prefix="atlantis_dqn_model"
    )
    
    print("Starting training...")
    model.learn(
        total_timesteps=10_000_000, 
        log_interval=10, 
        tb_log_name="DQN", 
        reset_num_timesteps=False,
        callback=checkpoint_callback
    )
    
    model_path = "atlantis_dqn_model"
    print(f"Saving model to {model_path}.zip...")
    model.save(model_path)
    
    print("Training complete! Model saved.")

if __name__ == "__main__":
    main()
