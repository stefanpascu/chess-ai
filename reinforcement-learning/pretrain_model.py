import chess
import chess.pgn
import torch
from torch.utils.data import Dataset, DataLoader
from chess_env import ChessEnv


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
        obs = self.fen_to_observation(fen)
        action = self.uci_to_action(move)
        return obs, action

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


# Pretraining is optional and can be replaced with self-play
def parse_pgn_to_dataset(pgn_file, output_file, max_games):
    with open(pgn_file) as pgn, open(output_file, "w") as out:
        for _ in range(max_games):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                out.write(f"{board.fen()},{move.uci()}\n")
                board.push(move)
