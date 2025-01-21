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


# Define Entropy Decay Callback
class EntropyDecayCallback(BaseCallback):
    def __init__(self, initial_entropy_coef, final_entropy_coef, total_timesteps, verbose=0):
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


def transfer_pretrained_weights_to_rl(pretrained_model_path, rl_model):
    """
    Transfers pretrained model weights to the RL model.

    Args:
        pretrained_model_path (str): Path to the pretrained model.
        rl_model (PPO): RL model instance.
    """
    print(f"Loading pretrained model from {pretrained_model_path}...")
    pretrained_model = PPO.load(pretrained_model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    rl_model.policy.load_state_dict(pretrained_model.policy.state_dict())
    print("Transferred pretrained weights to RL model.")


if __name__ == '__main__':
    num_envs = 4
    vec_env = SubprocVecEnv([make_chess_env for _ in range(num_envs)])
    vec_env = VecNormalize(vec_env)

    # Load existing VecNormalize stats if available
    if os.path.exists(settings.normalized_env_path):
        vec_env = VecNormalize.load(settings.normalized_env_path, vec_env)

    # Load or initialize the model
    pretrained_model_path = settings.pretrained_model_path  # Path to the pretrained model
    if os.path.exists(settings.reinforced_model_path):
        print(f"Loading and training existing model from {settings.reinforced_model_path}")
        model = PPO.load(settings.reinforced_model_path, env=vec_env, device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        user_input = input("An existing model was not found. \nDo you want to create a new model? (To continue 'yes'. Anything else will skip this step): ").strip().lower()
        if user_input == "yes":
            print("Creating and training new model...")
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                n_steps=4096,
                batch_size=1024,
                learning_rate=1e-4,  # Initial learning rate
                ent_coef=0.01,  # Initial entropy coefficient
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            if os.path.exists(pretrained_model_path):
                transfer_pretrained_weights_to_rl(pretrained_model_path, model)
        else:
            print("A new model was NOT created or trained because the user requested its cancellation.")
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

    # Train the model with callbacks
    model.learn(total_timesteps=total_timesteps, callback=[entropy_decay_callback, eval_callback, checkpoint_callback])

    # Save VecNormalize stats and the model
    vec_env.save(settings.normalized_env_path)
    model.save(settings.reinforced_model_path)

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print("Start time: " + str(start_time))
    print("End time: " + str(end_time))
    print("For a total of " + str(total_timesteps) + " time steps, the model was trained with a duration of: " + str(duration))

    # Save the final model
    model.save(settings.reinforced_model_path)
