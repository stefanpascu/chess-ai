import chess
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from chess_env import ChessEnv  # Ensure you're importing the class correctly
import torch
from stockfish import Stockfish  # Assuming Stockfish is installed and available for move quality

# Initialize Stockfish (optional for evaluating move quality)
stockfish = Stockfish(path="stockfish/stockfish-windows-x86-64-avx2.exe")


# Function to create chess environment
def make_chess_env():
    return ChessEnv(render_mode=None)  # No rendering during training for better efficiency


# Function to calculate rewards based on custom evaluation metrics
def evaluate_board(board, previous_board=None):
    reward = 0
    reward += evaluate_material(board)
    reward += evaluate_piece_activity(board)
    reward += evaluate_king_safety(board)

    if previous_board:
        reward += evaluate_move_quality(previous_board, board)  # Evaluate the quality of the move
    reward += evaluate_game_result(board)  # Additional rewards for checkmate/draw results

    return reward


# Material advantage (assign values to pieces)
def evaluate_material(board):
    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
    reward = 0
    for piece in board.pieces:
        if piece.is_white():
            reward += piece_values.get(piece.symbol().upper(), 0)
        else:
            reward -= piece_values.get(piece.symbol().upper(), 0)
    return reward


# Piece activity (reward for controlling center and moving pieces to active squares)
def evaluate_piece_activity(board):
    activity_reward = 0
    center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]  # d4, e4, d5, e5
    for piece in board.pieces:
        if piece.position in center_squares:
            activity_reward += 0.5  # Reward for central control
    return activity_reward


# King safety (reward for castling and keeping the king protected)
def evaluate_king_safety(board):
    king_safety = 0
    if board.has_castled():
        king_safety += 1  # Reward for castling
    return king_safety


# Move quality (compare the AI move to the best Stockfish move, optional)
def evaluate_move_quality(previous_board, current_board):
    stockfish.set_fen_position(previous_board.fen())
    best_move = stockfish.get_best_move()
    ai_move = current_board.last_move  # Assuming the ChessEnv class stores the last AI move
    if ai_move == best_move:
        return 1  # Perfect move
    else:
        return -1  # Sub-optimal move


# Endgame result (reward for winning and penalize for losing)
def evaluate_game_result(board):
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return 10  # White wins
        else:
            return -10  # Black wins
    elif board.is_stalemate() or board.is_draw():
        return 0  # Neutral reward for draw or stalemate
    return 0  # No terminal result


if __name__ == '__main__':  # Protect multiprocessing code on Windows
    num_envs = 4  # Number of parallel environments
    vec_env = SubprocVecEnv([make_chess_env for _ in range(num_envs)])

    # Normalize inputs to stabilize training
    vec_env = VecNormalize(vec_env)

    # Initialize the PPO model with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,  # Increase to consider longer sequences of steps
        batch_size=64,  # Larger batch size for stable updates
        learning_rate=3e-4,  # Default PPO learning rate
        device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    )

    # Configure logging for TensorBoard
    new_logger = configure("logs/", ["tensorboard"])
    model.set_logger(new_logger)

    # Set up evaluation and checkpoint callbacks
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=10000,  # Evaluate model every 10,000 steps
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./logs/',
        name_prefix='chess_model_checkpoint'
    )

    # Train the model with checkpoints and evaluation
    total_timesteps = 100000  # Increase total timesteps for better training
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

    # Save the final model
    model.save("chess_model")

    # To load the saved model later for playing a game, use:
    # model = PPO.load("chess_model", env=vec_env)

    # Load and play a game (rendering enabled for human-playable mode)
    vec_env = SubprocVecEnv([lambda: ChessEnv(render_mode=None)])  # Use 'human' mode for visual rendering
