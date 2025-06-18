import os
import time
import numpy as np
import chess
from stable_baselines3 import PPO
import minmax_negamax.negamax as ai_negamax
import reinforcement_learning.rl_settings as rl_settings

model_path = "reinforcement_learning//" + rl_settings.best_model_path if os.path.exists("reinforcement_learning//" + rl_settings.best_model_path) \
    else "reinforcement_learning//" + rl_settings.reinforced_model_path
engine = PPO.load(model_path)

NUM_GAMES = 10
MAX_MOVES_PER_GAME = 100

results = {
    "negamax_wins": 0,
    "rl_wins": 0,
    "draws": 0,
    "avg_time_negamax": [],
    "avg_time_rl": [],
    "good_moves_negamax": [],
    "good_moves_rl": [],
}

def get_observation_from_board(board):
    obs = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row, col = divmod(i, 8)
            layer = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                layer += 6
            obs[layer, row, col] = 1.0
    return obs

def decide_rl_move(board):
    obs = get_observation_from_board(board)
    obs = np.expand_dims(obs, axis=0)

    action_array, _ = engine.predict(obs)

    action_idx = int(action_array[0])

    legal_moves = list(board.legal_moves)
    selected = legal_moves[action_idx % len(legal_moves)]
    return selected


def play_game():
    board = chess.Board()
    move_times_negamax = []
    move_times_rl = []
    good_moves_negamax = 0
    good_moves_rl = 0

    for i in range(MAX_MOVES_PER_GAME):
        if board.is_game_over():
            break

        if board.turn == chess.WHITE:
            start = time.time()
            move = ai_negamax.find_best_move(board, list(board.legal_moves))
            duration = time.time() - start
            move_times_negamax.append(duration)
            good_moves_negamax += 1 if move in board.legal_moves else 0
        else:
            start = time.time()
            move = decide_rl_move(board)
            duration = time.time() - start
            move_times_rl.append(duration)
            good_moves_rl += 1 if move in board.legal_moves else 0

        board.push(move)

    if board.is_checkmate():
        if board.turn == chess.WHITE:
            results["negamax_wins"] += 1
        else:
            results["rl_wins"] += 1
    else:
        results["draws"] += 1

    if move_times_negamax:
        results["avg_time_negamax"].append(sum(move_times_negamax) / len(move_times_negamax))
    if move_times_rl:
        results["avg_time_rl"].append(sum(move_times_rl) / len(move_times_rl))
    results["good_moves_negamax"].append(good_moves_negamax)
    results["good_moves_rl"].append(good_moves_rl)

for _ in range(NUM_GAMES):
    play_game()

print("\n=== COMPARAȚIE AI ===")
print(f"Partide testate: {NUM_GAMES}")
print(f"Victorii Negamax: {results['negamax_wins']}")
print(f"Victorii RL: {results['rl_wins']}")
print(f"Remize: {results['draws']}")
print(f"Timp mediu decizie Negamax: {round(np.mean(results['avg_time_negamax']), 3)} sec")
print(f"Timp mediu decizie RL: {round(np.mean(results['avg_time_rl']), 3)} sec")
print(f"Mutări bune Negamax (medie): {round(np.mean(results['good_moves_negamax']), 2)}")
print(f"Mutări bune RL (medie): {round(np.mean(results['good_moves_rl']), 2)}")