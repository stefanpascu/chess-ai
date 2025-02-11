import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Debug GPU errors
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # Deterministic behavior

import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from chess_env import ChessEnv
import torch
import settings
import torch.nn as nn
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
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).view(sample_input.size(0), -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.fc(self.cnn(observations))


def make_chess_env():
    env = ChessEnv(reward_scaling_factor=1.0)
    env = Monitor(env)
    return env


def evaluate_models(new_model, best_model, num_games=10):
    """Play games between the new and best models to evaluate their performance."""
    wins, losses, draws = 0, 0, 0
    env = ChessEnv(reward_scaling_factor=0.01)  # Single-instance environment for evaluation

    for _ in range(num_games):
        env.reset()
        done = False
        while not done:
            # Alternate between the models
            if env.board.turn:
                action, _ = best_model.predict(env.get_observation(), deterministic=True)
            else:
                action, _ = new_model.predict(env.get_observation(), deterministic=True)
            _, reward, done, _, _ = env.step(action)

        # Update scores based on game result
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    return wins, losses, draws


if __name__ == "__main__":
    num_envs = settings.num_envs
    batch_size = settings.batch_size
    n_steps = settings.n_steps
    vec_env = SubprocVecEnv([make_chess_env for _ in range(num_envs)])
    vec_env = VecNormalize(vec_env)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "normalize_images": False,
        "use_sde": False,
        "clip_range_vf": None,
        "clip_range": 0.2,
        "torch_jit": True,  # Use TorchScript optimization if available
        "dtype": torch.float16,  # Mixed precision
    }

    model_path = settings.reinforced_model_path
    best_model_path = settings.best_model_path  # Path to save the best model
    total_timesteps = settings.training_number_of_timestamps

    # Load the best model or initialize a new one
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        best_model = PPO.load(
            best_model_path,
            device=device,
            n_steps=n_steps,
            batch_size=batch_size
        )
    else:
        print("No best model found. Creating a new one...")
        best_model = PPO(
            "CnnPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=1e-4,
            ent_coef=0.01,
            device=device,
        )

    # Initialize or load the current model
    if os.path.exists(model_path):
        print(f"Loading current model from {model_path}...")
        model = PPO.load(
            model_path,
            env=vec_env,
            device=device,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size
        )
    else:
        print("Creating a new PPO model...")
        model = PPO(
            "CnnPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=1e-4,
            ent_coef=0.01,
            device=device,
        )

    # Configure logger
    logger = configure("logs/", ["tensorboard"])
    model.set_logger(logger)

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./logs/",
        name_prefix="chess_model_checkpoint",
    )

    # Start training
    print(f"Model device: {model.policy.device}")
    start_time = datetime.datetime.now()
    print(f"Training started at {start_time}...")

    while True:
        # Train for a chunk of timesteps
        model.learn(
            total_timesteps=total_timesteps,  # Train in chunks of 200,000 timesteps
            callback=[eval_callback, checkpoint_callback],
        )

        # Evaluate the new model against the best
        print("Evaluating the new model against the best model...")
        wins, losses, draws = evaluate_models(model, best_model, num_games=100)
        print(f"Evaluation results: {wins} Wins, {losses} Losses, {draws} Draws")

        # Replace the best model if the new model performs better
        if wins > losses:
            print("New model outperformed the best model. Updating the best model...")
            model.save(best_model_path)
            best_model = PPO.load(best_model_path, device=device)
        else:
            print("Best model retained.")

        # Save the current model
        model.save(model_path)
        vec_env.save(settings.normalized_env_path)

        checkpoint_time = datetime.datetime.now()
        print(f"Training checkpoint at {checkpoint_time}...")
        print(f"Reaching this checkpoint took {checkpoint_time-start_time}...")