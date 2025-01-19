import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import chess
from stockfish import Stockfish
from stable_baselines3 import PPO
from chess_env import ChessEnv
import settings

# Initialize Stockfish as the intermediate opponent
stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"  # Update with your Stockfish executable path
stockfish = Stockfish(path=stockfish_path, parameters={"Skill Level": 1})  # Skill level 5 is intermediate

# Load the trained PPO model
model = PPO.load(settings.model_file_path)

# Initialize environment
env = ChessEnv()


def evaluate_model_against_stockfish(num_games=10):
    results = {"win": 0, "loss": 0, "draw": 0}
    for game_num in range(num_games):
        board = chess.Board()
        print(f"\nGame {game_num + 1}/{num_games}")

        while not board.is_game_over():
            if board.turn:  # White's turn (your AI)
                obs = env.get_observation()  # Get observation
                action, _ = model.predict(obs, deterministic=True)  # AI move
                legal_moves = list(board.legal_moves)
                move = legal_moves[action % len(legal_moves)]  # Ensure valid move
                board.push(move)
                env.board = board  # Sync environment with board state
            else:  # Black's turn (Stockfish)
                stockfish.set_fen_position(board.fen())
                stockfish_move = stockfish.get_best_move()
                if stockfish_move:
                    board.push(chess.Move.from_uci(stockfish_move))

            # print(board)

        # Evaluate the result
        if board.result() == "1-0":
            results["win"] += 1
            print("AI wins!")
        elif board.result() == "0-1":
            results["loss"] += 1
            print("Stockfish wins!")
        else:
            results["draw"] += 1
            print("Game drawn.")

    return results


# Run the evaluation
evaluation_results = evaluate_model_against_stockfish(num_games=10)
print("\nEvaluation Results:")
print(f"Wins: {evaluation_results['win']}, Losses: {evaluation_results['loss']}, Draws: {evaluation_results['draw']}")
