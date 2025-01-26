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
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
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


# Define Entropy Decay Callback
class EntropyDecayCallback(BaseCallback):
    def __init__(self, initial_entropy_coef, final_entropy_coef, total_timesteps, verbose=1):
        super(EntropyDecayCallback, self).__init__(verbose)
        self.initial_entropy_coef = initial_entropy_coef
        self.final_entropy_coef = final_entropy_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # Calculate the new entropy coefficient based on training progress
        progress = min(max(self.num_timesteps / self.total_timesteps, 0), 1)
        new_entropy_coef = (
            self.initial_entropy_coef +
            progress * (self.final_entropy_coef - self.initial_entropy_coef)
        )
        self.model.ent_coef = new_entropy_coef
        return True

# Environment setup
def make_chess_env():
    env = ChessEnv(reward_scaling_factor=0.0001)
    env = Monitor(env)
    return env

# Function to transfer weights from a pretrained model
def transfer_pretrained_weights(pretrained_model_path, rl_model):
    print(f"Loading pretrained model from {pretrained_model_path}...")
    pretrained_model = PPO.load(pretrained_model_path, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Selectively transfer compatible weights
    pretrained_state = pretrained_model.policy.state_dict()
    rl_state = rl_model.policy.state_dict()
    compatible_state = {k: v for k, v in pretrained_state.items() if k in rl_state and v.size() == rl_state[k].size()}
    rl_state.update(compatible_state)
    rl_model.policy.load_state_dict(rl_state)
    print(f"Transferred {len(compatible_state)}/{len(pretrained_state)} layers from the pretrained model.")

if __name__ == '__main__':
    num_envs = 4
    vec_env = SubprocVecEnv([make_chess_env for _ in range(num_envs)])
    vec_env = VecNormalize(vec_env)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 768},
        "normalize_images": False
    }

    pretrained_model_path = settings.pretrained_model_path  # Path to pretrained model
    reinforced_model_path = settings.reinforced_model_path  # Path to RL model
    total_timesteps = settings.training_number_of_timestamps

    if os.path.exists(reinforced_model_path):
        print(f"Loading and training existing RL model from {reinforced_model_path}...")
        model = PPO.load(reinforced_model_path, env=vec_env, device=device, verbose=1)
    else:
        user_input = input(f"RL model not found at {reinforced_model_path}. Load pretrained model? (yes/no): ").strip().lower()
        if user_input == "yes" and os.path.exists(pretrained_model_path):
            print("Initializing RL model from pretrained model...")
            model = PPO(
                "CnnPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                n_steps=4096,
                batch_size=1024,
                learning_rate=1e-4,
                ent_coef=0.01,
                device=device
            )
            transfer_pretrained_weights(pretrained_model_path, model)
        else:
            print("Creating a new RL model...")
            model = PPO(
                "CnnPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                n_steps=4096,
                batch_size=1024,
                learning_rate=1e-4,
                ent_coef=0.01,
                device=device
            )

    # Configure logger
    logger = configure("logs/", ["tensorboard"])
    model.set_logger(logger)

    # Define callbacks
    entropy_decay_callback = EntropyDecayCallback(
        initial_entropy_coef=0.1,
        final_entropy_coef=0.001,  # Lower entropy for exploitation
        total_timesteps=total_timesteps,
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

    # Start training
    start_time = datetime.datetime.now()
    print(f"Training started at {start_time}...")
    model.learn(total_timesteps=total_timesteps, callback=[entropy_decay_callback, eval_callback, checkpoint_callback])

    # Save the model and VecNormalize stats
    vec_env.save(settings.normalized_env_path)
    model.save(reinforced_model_path)

    end_time = datetime.datetime.now()
    print(f"Training completed at {end_time}. \nDuration: {end_time - start_time}")