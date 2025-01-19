import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from gym import spaces
import gym
import settings
import datetime


def parse_pgn_to_dataset(pgn_file, output_file, max_games):
    with open(pgn_file) as pgn, open(output_file, "w") as out:
        game_count = 0
        while game_count < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()
                out.write(f"{fen},{move.uci()}\n")
                board.push(move)
            game_count += 1
            if game_count % 1000 == 0:
                print(f"\rProcessed {game_count}/{settings.max_games_for_pretraining} games", end="")
    print(f"Finished processing {game_count} games. Dataset saved to {output_file}")


class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, "r") as f:
            for line in f:
                fen, move = line.strip().split(",")
                self.data.append((fen, move))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, move = self.data[idx]
        observation = self.fen_to_observation(fen)
        action = self.uci_to_action(move)
        return torch.tensor(observation, dtype=torch.float32), torch.tensor(action, dtype=torch.long)

    def fen_to_observation(self, fen):
        board = chess.Board(fen)
        obs = torch.zeros((12, 8, 8), dtype=torch.float32)
        piece_map = {
            "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
            "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11
        }
        for square, piece in board.piece_map().items():
            x, y = divmod(square, 8)
            obs[piece_map[piece.symbol()], x, y] = 1
        return obs

    def uci_to_action(self, uci):
        move = chess.Move.from_uci(uci)
        return move.from_square * 64 + move.to_square


def pretrain_model(model, dataset, epochs=10, batch_size=64, lr=1e-3, num_workers=0):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.policy.parameters(), lr=lr)
    device = next(model.policy.parameters()).device

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}...")
        epoch_start_time = time.time()
        batch_start_time = time.time()
        total_loss = 0
        for batch_idx, (obs, actions) in enumerate(dataloader):
            obs = obs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True).long()
            obs = obs.view(obs.size(0), -1)

            optimizer.zero_grad()
            policy_features, _ = model.policy.mlp_extractor(obs)
            logits = model.policy.action_net(policy_features)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 1000 == 0:
                batch_time = time.time() - batch_start_time
                print(f"\rBatch {batch_idx + 1}/{len(dataloader)} processed in {batch_time:.2f}s - Loss: {loss.item():.4f}", end="")
                batch_start_time = time.time()

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s. Average Loss: {total_loss / len(dataloader):.4f}")


class DummyChessEnv(gym.Env):
    def __init__(self):
        super(DummyChessEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(12, 8, 8), dtype=float)
        self.action_space = spaces.Discrete(4672)

    def step(self, action):
        return self.observation_space.sample(), 0, True, {}

    def reset(self):
        return self.observation_space.sample()


def main():
    pgn_file = settings.pretraining_pgn_data
    output_file = "parsed_chess_dataset.csv"

    user_input = input("Do you want to parse the PGN file? (yes/no): ").strip().lower()
    if user_input == "yes":
        start_time = datetime.datetime.now()
        print(f"Extracting data from {pgn_file}...")
        max_games = settings.max_games_for_pretraining
        parse_pgn_to_dataset(pgn_file, output_file, max_games)
        end_time = datetime.datetime.now()
        print(f"Parsing took {end_time-start_time}.")
    else:
        print("Skipping PGN parsing...")

    # Load the dataset
    print("Loading dataset...")
    dataset = ChessDataset(output_file)
    print("Initializing model with a dummy environment...")
    dummy_env = DummyVecEnv([lambda: DummyChessEnv()])
    vec_env = VecNormalize(dummy_env)

    user_input = input("PRETRAINING the model will DELETE its current version. \nDo you want to continue and pretrain the model? (yes/no): ").strip().lower()
    if user_input == "yes":
        # Pretrain the model
        print("Starting pretraining...")
        start_time = datetime.datetime.now()
        model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda")
        pretrain_model(model, dataset, epochs=5, batch_size=64, lr=1e-3)
        end_time = datetime.datetime.now()
        print(f"Pretraining took: {end_time - start_time}.")
    else:
        print("A new model was NOT pretrained because the user requested its cancellation.")

    vec_env.save("vec_normalize.pkl")
    model.save(settings.model_file_path)
    print(f"Model and VecNormalize stats saved to {settings.model_file_path}")


if __name__ == "__main__":
    main()
