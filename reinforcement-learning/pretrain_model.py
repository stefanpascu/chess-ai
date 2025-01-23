import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gym import spaces
import gym
import settings
import datetime
import train_and_update_model as train_model
import stockfish_for_pretraining


class ChessDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the dataset CSV file.
        """
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
        # Ensure observation is a tensor and action is directly converted
        return torch.tensor(observation, dtype=torch.float32), torch.tensor(action, dtype=torch.long)

    def fen_to_observation(self, fen):
        # Convert FEN to a tensor-based observation (12x8x8 format)
        board = chess.Board(fen)
        obs = torch.zeros((12, 8, 8), dtype=torch.float32)
        piece_map = {
            "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,  # White pieces
            "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11  # Black pieces
        }
        for square, piece in board.piece_map().items():
            x, y = divmod(square, 8)
            obs[piece_map[piece.symbol()], x, y] = 1
        return obs

    def uci_to_action(self, uci):
        # Map UCI move to a discrete action space index
        move = chess.Move.from_uci(uci)
        return move.from_square * 64 + move.to_square


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


def pretrain_model_with_entropy_and_stochastic_sampling(
    model, dataset, epochs=10, batch_size=64, lr=1e-3, entropy_coef=0.01, temperature=1.0, num_workers=0
):
    """
    Pretrain the model with Stockfish evaluation, entropy regularization, and stochastic action sampling.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    criterion_move = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.policy.parameters(), lr=lr)
    device = next(model.policy.parameters()).device

    for epoch in range(epochs):
        initial_temperature = temperature
        final_temperature = temperature / 10
        temperature = initial_temperature - (epoch / epochs) * (initial_temperature - final_temperature)

        print(f"Starting epoch {epoch + 1}/{epochs}...")
        epoch_start_time = time.time()
        batch_start_time = time.time()
        total_loss = 0
        batch_id = 0

        print(f"\rBatches 0/{len(dataloader)}", end="")
        for obs, actions in dataloader:
            obs = obs.to(device, non_blocking=True).view(obs.size(0), -1)  # Flatten observations
            actions = actions.to(device, non_blocking=True).long()

            optimizer.zero_grad()

            policy_features, _ = model.policy.mlp_extractor(obs)
            logits = model.policy.action_net(policy_features)

            # Stochastic action sampling
            sampled_actions = stockfish_for_pretraining.sample_stochastic_action(logits, temperature=temperature)

            # Calculate losses
            move_loss = criterion_move(logits, actions)
            sampled_actions_loss = criterion_move(logits, sampled_actions.view(-1))
            entropy_loss = -entropy_coef * (logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1).mean()

            loss = move_loss + 0.1 * sampled_actions_loss + entropy_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_id + 1) % (settings.max_games_for_pretraining//100) == 0:
                batch_time = time.time() - batch_start_time
                print(f"\rBatches {batch_id + 1}/{len(dataloader)} processed in {batch_time:.2f}s - Loss: {loss.item():.4f}", end="")
                batch_start_time = time.time()

            batch_id += 1

        print(f"\rBatches {len(dataloader)}/{len(dataloader)} - Loss: {loss.item():.4f}", end="")
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s. Avg Loss: {total_loss / len(dataloader):.4f}")


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
    parsed_data_file = settings.parsed_pgn_data
    max_games = settings.max_games_for_pretraining
    is_stockfish = input("Do you want to use stockfish for pretraining? (yes/no): ").strip().lower()
    stockfish_path = settings.stockfish_path
    policy_kwargs = {
        "features_extractor_class": train_model.CustomCNN,
        "features_extractor_kwargs": {"features_dim": 256},  # Size of the final feature vector
        "normalize_images": False  # Already normalized in the environment
    }
    if is_stockfish == "yes":
        print(f"Stockfish enabled.")
    else:
        print(f"Stockfish disabled.")

    user_input = input("Do you want to parse the PGN file? (yes/no): ").strip().lower()
    if user_input == "yes":
        start_time = datetime.datetime.now()
        print(f"Extracting data from {pgn_file}...")
        if is_stockfish == "yes":
            stockfish_for_pretraining.parse_pgn_to_dataset_with_stockfish_optimized(pgn_file, parsed_data_file, max_games, stockfish_path)
        else:
            parse_pgn_to_dataset(pgn_file, parsed_data_file, max_games)
        end_time = datetime.datetime.now()
        print(f"Parsing took {end_time - start_time}.")
    else:
        print("Skipping PGN parsing...")

    print("Loading dataset...")
    if is_stockfish == "yes":
        dataset = stockfish_for_pretraining.ChessDatasetWithStockfish(parsed_data_file)
    else:
        dataset = ChessDataset(parsed_data_file)
    print("Initializing model...")
    dummy_env = DummyVecEnv([lambda: DummyChessEnv()])
    vec_env = VecNormalize(dummy_env)

    user_input = input(
        "PRETRAINING the model will DELETE its current version. \nDo you want to continue? (yes/no): ").strip().lower()
    if user_input == "yes":
        # Pretrain the model
        start_time = datetime.datetime.now()

        model = PPO(
            "CnnPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            device="cuda",
            verbose=1
        )

        print("Starting pretraining...")
        if is_stockfish == "yes":
            stockfish_for_pretraining.pretrain_model_with_stockfish_and_entropy_and_stochastic_sampling(model, dataset, epochs=5, batch_size=64, lr=1e-3)
        else:
            pretrain_model_with_entropy_and_stochastic_sampling(model, dataset, epochs=5, batch_size=64, lr=1e-3)
        vec_env.save(settings.normalized_env_path)
        model.save(settings.pretrained_model_path)
        end_time = datetime.datetime.now()
        print(f"Pretraining took: {end_time - start_time}.")
        print(f"Model saved to {settings.pretrained_model_path}")
    else:
        print("Pretraining canceled.")


if __name__ == "__main__":
    main()