import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    parser = argparse.ArgumentParser(description="Test trained Atlantis DQN agent.")
    parser.add_argument("--render", action="store_true", help="Render the environment visually")
    args = parser.parse_args()

    env_id = "ALE/Atlantis-v5"
    print("Loading environment...")
    
    # If rendering is enabled, we pass render_mode="human" to the underlying environment
    env_kwargs = {}
    if args.render:
        env_kwargs["render_mode"] = "human"

    env = make_atari_env(env_id, n_envs=1, seed=0, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)

    model_path = "atlantis_dqn_model.zip"
    print(f"Loading model from {model_path}...")
    try:
        model = DQN.load(model_path, env=env)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Make sure you train it first.")
        return

    print("Evaluating model...")
    # Evaluate policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=args.render)
    
    print(f"Evaluation complete.")
    print(f"Mean Reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    main()
