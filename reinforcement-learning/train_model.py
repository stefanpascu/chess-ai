import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import datetime
from sb3_contrib import MaskablePPO  # For action masking support
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from chess_env import ChessEnv
import torch
import settings
import torch.nn as nn
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from wrappers import ActionMaskWrapper

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # Call the parent constructor to correctly set features_dim
        super(CustomCNN, self).__init__(observation_space, features_dim)

        board_shape = observation_space.spaces["observation"].shape  # e.g. (18, 8, 8)
        n_input_channels = board_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the flattened dimension
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.spaces["observation"].sample()[None]).float()
            n_flatten = self.cnn(sample_input).view(sample_input.size(0), -1).shape[1]

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        board_obs = observations["observation"]
        return self.fc(self.cnn(board_obs))


class CustomMaskablePPO(MaskablePPO):
    def __init__(self, policy, env, *args, **kwargs):
        super().__init__(policy, env, *args, **kwargs)
        self.env = env

    def get_chess_env(self):
        """Recursively unwraps the environment to get the base ChessEnv instance."""
        env = self.env

        # Handle vectorized environments
        if hasattr(env, "envs"):  # DummyVecEnv, SubprocVecEnv
            env = env.envs[0]  # Get the first environment in the vector

        # Unwrap any other wrappers (e.g., ActionMaskWrapper)
        while hasattr(env, "env") and not hasattr(env, "get_action_mask"):
            env = env.env

        return env

    def predict(self, observation, state=None, deterministic=False, **kwargs):
        # if isinstance(observation, dict) and "episode_start" in observation:
        #     observation = {k: v for k, v in observation.items() if k != "episode_start"}

        if "action_mask" not in observation:
            raise RuntimeError("ERROR: 'action_mask' is missing from observation!")

        # action_mask = observation["action_mask"]
        # action_mask = self.env.get_action_mask()
        chess_env = self.get_chess_env()

        # Fetch the action mask properly
        action_mask = np.array(chess_env.get_action_mask(), dtype=bool)

        if np.all(action_mask == 0):
            print("🚨 Game is over, but predict() was called! Skipping prediction.")
            return None, None

        # print(f"DEBUG: Extracted action mask: {action_mask}")

        if not np.array_equal(action_mask, action_mask.astype(bool)):
            raise RuntimeError("🚨 ERROR: Action mask contains non-binary values: ", action_mask)

        actions, _ = super().predict(observation, state=state, deterministic=deterministic, action_masks=action_mask, **kwargs)

        return actions, _


# --- Environment Creation Function ---
def make_chess_env():
    env = ChessEnv(reward_scaling_factor=1.0)
    env = Monitor(env)
    env = ActionMaskWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def evaluate_models(new_model, best_model, num_games=10):
    """Evaluates two models by playing a set number of games against each other."""
    wins, losses, draws = 0, 0, 0

    env = ChessEnv(reward_scaling_factor=1.0)
    env = Monitor(env)
    env = ActionMaskWrapper(env)

    # Unwrap to access the underlying ChessEnv
    unwrapped_env = env.unwrapped

    for _ in range(num_games):
        obs, _ = env.reset()
        done = False

        while not done:
            if isinstance(obs, dict) and "episode_start" in obs:
                del obs["episode_start"]

            model = best_model if unwrapped_env.board.turn else new_model

            # Ensure action mask is in the observation
            if "action_mask" not in obs:
                obs["action_mask"] = np.array(env.action_masks(), dtype=bool)

            # 🚨 Ensure mask consistency
            if np.sum(obs["action_mask"]) == 0:
                print(f"🚨 ERROR: No valid actions available! Skipping turn.")
                break

            # 🚀 Debug: Print legal moves before choosing
            legal_moves = [unwrapped_env.all_possible_moves[i] for i, m in enumerate(obs["action_mask"]) if m]
            print(f"🔹 Legal moves: {legal_moves}")

            # 🚀 Call model.predict() without 'action_masks' argument (it is inside `obs`)
            action, _ = model.predict(obs, state=None, deterministic=True)

            # 🚀 Debug: Print chosen move
            chosen_move = unwrapped_env.all_possible_moves[action]
            print(f"🛠️ Model chose: {chosen_move}")

            if not obs["action_mask"][action]:
                print(f"🚨 ERROR: Chosen move {chosen_move} is illegal!")
                raise RuntimeError(f"Illegal move selected: {chosen_move}")

            obs, reward, done, _, _ = env.step(action)

        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    return wins, losses, draws


if __name__ == "__main__":
    num_envs = 4
    # vec_env = SubprocVecEnv([make_chess_env for _ in range(num_envs)])
    vec_env = make_chess_env()
    # vec_env = VecNormalize(vec_env)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = "MultiInputPolicy"

    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "normalize_images": False,
    }

    model_path = settings.reinforced_model_path
    best_model_path = settings.best_model_path
    total_timesteps = settings.training_number_of_timestamps

    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        best_model = CustomMaskablePPO.load(best_model_path, device=device)
    else:
        print("No best model found. Creating a new one...")
        best_model = CustomMaskablePPO(
            policy,
            vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=4096,
            batch_size=1024,
            learning_rate=1e-4,
            ent_coef=0.01,
            device=device,
        )

    if os.path.exists(model_path):
        print(f"Loading current model from {model_path}...")
        model = CustomMaskablePPO.load(model_path, env=vec_env, device=device, verbose=1)
    else:
        print("Creating a new PPO model...")
        model = CustomMaskablePPO(
            policy,
            vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=4096,
            batch_size=1024,
            learning_rate=1e-4,
            ent_coef=0.01,
            device=device,
        )

    logger = configure("logs/", ["tensorboard"])
    model.set_logger(logger)

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=16_384,
        deterministic=True,
        render=False,
    )

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=100000,
    #     save_path="./logs/",
    #     name_prefix="chess_model_checkpoint",
    # )

    start_time = datetime.datetime.now()
    print(f"Training started at {start_time}...")

    while True:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[
                eval_callback,
                # checkpoint_callback
            ],
        )

        print("Evaluating the new model against the best model...")
        # wins, losses, draws = evaluate_models(model, best_model, num_games=20)
        wins, losses, draws = 0, 0, 20 # TODO: evaluate_models() does not work. MAKE IT WORK
        print(f"Evaluation results: {wins} Wins, {losses} Losses, {draws} Draws")

        if wins > losses:
            print("New model outperformed the best model. Updating the best model...")
            model.save(best_model_path)
            best_model = CustomMaskablePPO.load(best_model_path, device=device)
        else:
            print("Best model retained.")

        model.save(model_path)

        checkpoint_time = datetime.datetime.now()
        print(f"Training checkpoint at {checkpoint_time}...")
        print(f"Training duration since start: {checkpoint_time - start_time}...")

        if isinstance(vec_env, VecNormalize):
            vec_env.save(settings.normalized_env_path)
        else:
            print("⚠️ Skipping vec_env saving: Only VecNormalize environments can be saved.")
