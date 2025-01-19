import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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


# Step 1: Parse PGN file to create a dataset
def parse_pgn_to_dataset(pgn_file, output_file, max_games):
    """
    Parses a PGN file and writes FEN positions with corresponding moves to a file.
    Args:
        pgn_file (str): Path to the PGN file.
        output_file (str): Path to save the parsed dataset.
        max_games (int): Number of games to process.
    """
    with open(pgn_file) as pgn, open(output_file, "w") as out:
        game_count = 0
        while game_count < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break  # End of PGN file
            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()
                out.write(f"{fen},{move.uci()}\n")
                board.push(move)
            game_count += 1
            if game_count % 1000 == 0:
                print(f"Processed {game_count}/{settings.max_games_for_pretraining} games")
    print(f"Finished processing {game_count} games. Dataset saved to {output_file}")


# Step 2: Define PyTorch dataset for supervised learning
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


# Step 3: Define pretraining function
import time

def pretrain_model(model, dataset, epochs=10, batch_size=64, lr=1e-3, num_workers=4):
    """
    Pretrain the model's policy network using supervised learning.
    Args:
        model (PPO): The PPO model.
        dataset (ChessDataset): The training dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        num_workers (int): Number of workers for DataLoader to parallelize data loading.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Parallelize data loading
        pin_memory=True          # Optimize data transfer to GPU
    )
    criterion = nn.CrossEntropyLoss()  # For supervised classification
    optimizer = optim.Adam(model.policy.parameters(), lr=lr)  # Access the policy's parameters

    # Get the device of the policy network
    device = next(model.policy.parameters()).device

    for epoch in range(epochs):
        total_loss = 0
        print(f"Starting epoch {epoch + 1}/{epochs}...")
        epoch_start_time = time.time()

        batch_start_time = time.time()
        for batch_idx, (obs, actions) in enumerate(dataloader):

            # Move observations and actions to the same device as the model
            obs = obs.to(device, non_blocking=True)  # Enable non-blocking transfer for speed
            actions = actions.to(device, non_blocking=True).long()

            # Flatten the observations
            obs = obs.view(obs.size(0), -1)

            optimizer.zero_grad()

            # Forward pass: Extract policy features and logits
            policy_features, _ = model.policy.mlp_extractor(obs)  # Extract policy and value features
            logits = model.policy.action_net(policy_features)  # Actor head outputs logits

            # Compute loss
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log batch processing time
            if (batch_idx + 1) % 1000 == 0:
                batch_time = time.time() - batch_start_time
                print(f"Batch {batch_idx + 1}/{len(dataloader)} processed in {batch_time:.2f}s - Loss: {loss.item():.4f}")
                batch_start_time = time.time()

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s. Average Loss: {total_loss / len(dataloader):.4f}")


# Step 4: Dummy environment for PPO initialization
class DummyChessEnv(gym.Env):
    def __init__(self):
        super(DummyChessEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(12, 8, 8), dtype=float)
        self.action_space = spaces.Discrete(4672)

    def step(self, action):
        return self.observation_space.sample(), 0, True, {}

    def reset(self):
        return self.observation_space.sample()

# Step 5: Combine everything
def main():
    pgn_file = "pretraining-data/lichess_db_standard_rated_2024-12.pgn/lichess_db_standard_rated_2024-12.pgn"  # Replace with your PGN file path
    output_file = "parsed_chess_dataset.csv"

    # Ask user if they want to parse the PGN file
    user_input = input("Do you want to parse the PGN file? \n(To parse the file type 'yes'. Anything else will skip this step): ").strip().lower()
    if user_input == "yes":
        # Prompt the user for max_games
        max_games = settings.max_games_for_pretraining  # Default value if input is invalid
        print(f"Maximum games set to: {max_games}")

        print("Parsing PGN file...")
        parse_pgn_to_dataset(pgn_file, output_file, max_games)
    else:
        print("Skipping PGN parsing...")

    # Load the dataset
    print("Loading dataset...")
    dataset = ChessDataset(output_file)

    # Create a dummy environment for PPO initialization
    print("Initializing model with a dummy environment...")
    dummy_env = DummyChessEnv()
    model = PPO("MlpPolicy", dummy_env, verbose=1, device="cuda")

    user_input = input("Do you want to pretrain the model? \nThis action will delete the existing model and create a new one from scratch. \n(To continue type 'yes'. Anything else will skip this step): ").strip().lower()
    if user_input == "yes":
        # Pretrain the model
        print("Starting pretraining...")
        pretrain_model(model, dataset, epochs=5, batch_size=64, lr=1e-3)

        # Save the pretrained model
        model.save(settings.model_file_path)
        print(f"Model saved as {settings.model_file_path}")
    else:
        print("A new model was NOT pretrained because the user requested its cancellation.")


if __name__ == "__main__":
    main()
