import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from chess_env import ChessEnv
import torch
import settings
import sys

def make_chess_env():
    return ChessEnv()

if __name__ == '__main__':
    num_envs = 4
    vec_env = SubprocVecEnv([make_chess_env for _ in range(num_envs)])
    vec_env = VecNormalize(vec_env)

    if os.path.exists("vec_normalize.pkl"):
        vec_env = VecNormalize.load("vec_normalize.pkl", vec_env)

    if os.path.exists(settings.model_file_path):
        print(f"Loading and training existing model from {settings.model_file_path}")
        model = PPO.load(settings.model_file_path, env=vec_env, device='cuda' if torch.cuda.is_available() else 'cpu')
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
                learning_rate=1e-4, # 3e-4,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            print("A new model was NOT created or trained because the user requested its cancellation.")
            sys.exit(0)

    new_logger = configure("logs/", ["tensorboard"])
    model.set_logger(new_logger)

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./logs/',
        name_prefix='chess_model_checkpoint'
    )

    total_timesteps = settings.training_number_of_timestamps

    start_time = datetime.datetime.now()

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

    vec_env.save("vec_normalize.pkl")
    model.save(settings.model_file_path)

    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print("Start time: " + str(start_time))
    print("End time: " + str(end_time))
    print("For a total of " + str(total_timesteps) + " time steps, the model was trained with a duration of: " + str(duration))

    # Save the final model
    model.save(settings.model_file_path)

