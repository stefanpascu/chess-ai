import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from chess_env import ChessEnv
import torch
import settings
import sys
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute the number of features output by the CNN
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).view(sample_input.size(0), -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.fc(self.cnn(observations))


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, log_interval=settings.training_number_of_timestamps//100, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_counter = 0

    def _on_step(self) -> bool:
        # Check if episode information is available
        if len(self.model.ep_info_buffer) > 0:
            # Increment the episode counter
            self.episode_counter += 1

            # Log every 'log_interval' episodes
            if self.episode_counter % self.log_interval == 0:
                # Calculate the average reward over the episodes
                avg_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                if np.isfinite(avg_reward):
                    print(f"Episode: {self.episode_counter}, Avg Reward: {avg_reward:.2f}")
                else:
                    print(f"Episode: {self.episode_counter}, Avg Reward: NaN (invalid rewards in buffer)")
        return True


# Define Entropy Decay Callback
class EntropyDecayCallback(BaseCallback):
    def __init__(self, initial_entropy_coef, final_entropy_coef, total_timesteps, verbose=1):
        super(EntropyDecayCallback, self).__init__(verbose)
        self.initial_entropy_coef = initial_entropy_coef
        self.final_entropy_coef = final_entropy_coef
        self.total_timesteps = total_timesteps


    def _on_step(self) -> bool:
        # Calculate the new entropy coefficient based on training progress
        progress = self.num_timesteps / self.total_timesteps
        new_entropy_coef = (
            self.initial_entropy_coef +
            progress * (self.final_entropy_coef - self.initial_entropy_coef)
        )
        self.model.ent_coef = new_entropy_coef
        return True


# Environment setup
def make_chess_env():
    return ChessEnv()


def transfer_pretrained_weights_to_rl(pretrained_model_path, rl_model, vec_env, normalize_stats_path):
    print(f"Loading pretrained model from {pretrained_model_path}...")

    # Load the pretrained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_model = PPO.load(pretrained_model_path, device=device)

    # Selectively transfer compatible weights
    pretrained_state = pretrained_model.policy.state_dict()
    rl_state = rl_model.policy.state_dict()
    compatible_state = {k: v for k, v in pretrained_state.items() if k in rl_state and v.size() == rl_state[k].size()}
    rl_state.update(compatible_state)
    rl_model.policy.load_state_dict(rl_state)
    print(f"Transferred {len(compatible_state)}/{len(pretrained_state)} layers from pretrained model to RL model.")

    # Ensure VecNormalize stats are consistent
    if os.path.exists(normalize_stats_path):
        print(f"Loading VecNormalize stats from {normalize_stats_path}...")
        vec_env = VecNormalize.load(normalize_stats_path, vec_env)
        vec_env.training = True  # Enable training mode
        vec_env.norm_reward = True  # Normalize rewards
        print("VecNormalize stats loaded and environment updated.")
    else:
        print("No VecNormalize stats found. Continuing with a new environment.")

    return vec_env


if __name__ == '__main__':
    num_envs = 4
    vec_env = SubprocVecEnv([make_chess_env for _ in range(num_envs)])
    vec_env = VecNormalize(vec_env)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = "CnnPolicy" # "MlpPolicy"
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 768},  # Adjust this if needed
        "normalize_images": False
    }

    # Load existing VecNormalize stats if available
    if os.path.exists(settings.normalized_env_path):
        vec_env = VecNormalize.load(settings.normalized_env_path, vec_env)

    # Load or initialize the model
    pretrained_model_path = settings.pretrained_model_path  # Path to the pretrained model
    if os.path.exists(settings.reinforced_model_path):
        print(f"Loading and training existing reinforcement learning model from {settings.reinforced_model_path}")
        model = PPO.load(settings.reinforced_model_path, env=vec_env, verbose=1, device=device)
    else:
        user_input = input(f"\nReinforcement learning model from {settings.reinforced_model_path} was not found. \nDo you want to load and train pretrained model? (yes/no): ").strip().lower()
        if user_input == "yes":
            if os.path.exists(pretrained_model_path):
                print(f"Loading and training existing pretrained model from {pretrained_model_path}...")
                model = PPO.load(settings.pretrained_model_path, env=vec_env, verbose=1,
                                 device=device)

            else:
                user_input = input("\nNo existing model found. \nDo you want to create a new model? (yes/no): ").strip().lower()
                if user_input == "yes":
                    print("Creating and training new model...")
                    model = PPO(
                        policy,
                        vec_env,
                        verbose=1,
                        n_steps=4096,
                        batch_size=1024,
                        learning_rate=1e-4,  # Initial learning rate
                        ent_coef=0.01,  # Initial entropy coefficient
                        device=device,
                        policy_kwargs=policy_kwargs
                    )

                else:
                    print("Model creation was cancelled.")
                    sys.exit(0)
        else:
            user_input = input(
                "\nDo you want to create a new model? (yes/no): ").strip().lower()
            if user_input == "yes":
                print("Creating and training new model...")
                model = PPO(
                    policy,
                    vec_env,
                    verbose=1,
                    n_steps=4096,
                    batch_size=1024,
                    learning_rate=1e-4,  # Initial learning rate
                    ent_coef=0.01,  # Initial entropy coefficient
                    device=device,
                    policy_kwargs=policy_kwargs
                )
            else:
                print("Model creation was cancelled.")
                sys.exit(0)

    # Configure logger
    new_logger = configure("logs/", ["tensorboard"])
    model.set_logger(new_logger)

    # Define callbacks
    entropy_decay_callback = EntropyDecayCallback(
        initial_entropy_coef=0.01,
        final_entropy_coef=0.001,  # Lower entropy for exploitation
        total_timesteps=settings.training_number_of_timestamps,
    )

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./logs/',
        name_prefix='chess_model_checkpoint',
    )

    total_timesteps = settings.training_number_of_timestamps

    start_time = datetime.datetime.now()
    print("Start time: " + str(start_time))

    # Train the model with callbacks
    model.learn(total_timesteps=total_timesteps, callback=[entropy_decay_callback, eval_callback, checkpoint_callback])

    # Save VecNormalize stats and the model
    vec_env.save(settings.normalized_env_path)
    model.save(settings.reinforced_model_path)

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print("End time: " + str(end_time))
    print("For a total of " + str(total_timesteps) + " time steps, the model was trained with a duration of: " + str(duration))

    # Save the final model
    model.save(settings.reinforced_model_path)
