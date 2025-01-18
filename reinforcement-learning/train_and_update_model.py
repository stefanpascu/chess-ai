import datetime

import chess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from chess_env import ChessEnv  # Ensure you're importing the class correctly
import torch

# Function to create chess environment
def make_chess_env():
    return ChessEnv()  # No rendering during training for better efficiency

if __name__ == '__main__':  # Protect multiprocessing code on Windows
    num_envs = 4  # Number of parallel environments
    vec_env = SubprocVecEnv([make_chess_env for _ in range(num_envs)])

    # Normalize inputs to stabilize training
    vec_env = VecNormalize(vec_env)

    # Initialize the PPO model with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,  # Increase to consider longer sequences of steps
        batch_size=64,  # Larger batch size for stable updates
        learning_rate=3e-4,  # Default PPO learning rate
        device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    )

    # Configure logging for TensorBoard
    new_logger = configure("logs/", ["tensorboard"])
    model.set_logger(new_logger)

    # Set up evaluation and checkpoint callbacks
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=10000,  # Evaluate model every 10,000 steps
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./logs/',
        name_prefix='chess_model_checkpoint'
    )

    # Train the model with checkpoints and evaluation
    total_timesteps = 10000  # Increase total timesteps for better training
    # total_timesteps = 1000000  # Increase total timesteps for better training ->1000000 for 0:20:54.877983
    # total_timesteps = 17000000  # Increase total timesteps for better training -> 17000000 for 6:42:10.494444

    start_time = datetime.datetime.now()

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print("Start time: " + str(start_time))
    print("End time: " + str(end_time))
    print("For a total of " + str(total_timesteps) + " time steps, the model was trained with a duration of: " + str(duration))

    # Save the final model
    model.save("chess_model")

    # To load the saved model later for playing a game, use:
    # model = PPO.load("chess_model", env=vec_env)

    # Load and play a game (rendering enabled for human-playable mode)
    vec_env = SubprocVecEnv([lambda: ChessEnv(render_mode=None)])  # Use 'human' mode for visual rendering



# Recommendations:
# 1. Tune Hyperparameters: Since rewards are fluctuating, it might help to:
#  - Decrease the learning rate to encourage more stable learning in later stages.
#  - Increase exploration by adjusting exploration-related parameters (e.g., epsilon decay for epsilon-greedy strategies).
# 2. Longer Training: Training over more timesteps (e.g., up to 500,000 or even 1,000,000) might help the agent converge to a more stable policy and continue improving.
# 3. Adjust Reward Structure: If the reward design focuses too much on intermediate moves rather than the endgame, you may want to adjust it to encourage better long-term strategy.
# 4. Early Stopping: If you see rewards peaking around 240,000 timesteps and not improving afterward, you could implement early stopping or a checkpoint system to evaluate the best policy found so far.