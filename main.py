import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def main():
    print("Setting up the Atlantis environment...")
    
    env_id = "ALE/Atlantis-v5"
    num_envs = 1
    
    env = make_atari_env(env_id, n_envs=num_envs, seed=0)
    
    # Frame-stacking with 4 frames to capture velocity
    env = VecFrameStack(env, n_stack=4)
    
    print("Initializing DQN agent...")
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
    
    print("Starting training...")
    model.learn(total_timesteps=10_000_000, log_interval=10, tb_log_name="DQN")
    
    model_path = "atlantis_dqn_model"
    print(f"Saving model to {model_path}.zip...")
    model.save(model_path)
    
    print("Training complete! Model saved.")

if __name__ == "__main__":
    main()
