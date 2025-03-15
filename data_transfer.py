import os
import serial
import time
import chess
import numpy as np
from stable_baselines3 import PPO
from reinforcement_learning import settings
from reinforcement_learning.chess_env import ChessEnv  # Importă mediul tău de șah personalizat

# Set up Serial communication with Arduino (actualizează portul și folosește același baud rate ca în Arduino: 9600)
arduino = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)  # Așteaptă pentru stabilizarea conexiunii
test_boolean = True

# Încarcă modelul AI (PPO) de șah
model_path = os.path.join("reinforcement_learning", settings.best_model_path) if os.path.exists(
    os.path.join("reinforcement_learning", settings.best_model_path)
) else os.path.join("reinforcement_learning", settings.reinforced_model_path)

engine = PPO.load(model_path)

# Inițializează mediul de șah și tabla
env = ChessEnv()
board = chess.Board()


def get_observation_from_board(board):
    """Convertește starea tablei de șah într-un format (12,8,8) compatibil cu PPO."""
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


def decide_move(board):
    """Folosește modelul PPO pentru a prezice cea mai bună mișcare de șah."""
    observation = get_observation_from_board(board)  # Obține observația de formă (12, 8, 8)
    observation = np.expand_dims(observation, axis=0)  # Adaugă dimensiunea de lot → (1, 12, 8, 8)
    action, _ = engine.predict(observation)  # Obține acțiunea de la model
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        return None  # Jocul s-a terminat
    action = int(action.item())  # Extrage un scalar Python din array-ul NumPy
    selected_move = legal_moves[action % len(legal_moves)]  # Maparea la o mișcare legală
    return selected_move


def map_move_to_servos(move):
    """Mapează o mișcare de șah către comenzi pentru servomotoare."""
    move_str = str(move)
    print(f"Mapping move: {move_str}")
    piece_mapping = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5
    }
    piece = board.piece_at(move.from_square)
    if piece is None:
        return None, None
    servo_index = piece_mapping.get(piece.symbol().upper(), 0)
    angle = 90  # Unghi default – ajustează logica dacă este necesar
    return servo_index, angle


def send_move_to_arduino(servo_index, angle):
    global test_boolean
    """Trimite comanda pentru servomotor către Arduino și așteaptă confirmarea."""
    if servo_index is None or angle is None:
        print("Invalid move, skipping Arduino command.")
        return

    # Alternăm pentru test între două comenzi (exemplu)
    if test_boolean:
        command = f"{servo_index},{90}\n"
        test_boolean = False
    else:
        command = f"{servo_index},{60}\n"
        test_boolean = True

    # Ștergem bufferul de intrare înainte de trimitere
    arduino.reset_input_buffer()

    arduino.write(command.encode())
    print(f"Sent command to Arduino: {command}")

    # Așteaptă confirmarea de la Arduino ("MOVE_DONE")
    response = ""
    timeout = time.time() + 10  # Timeout de 10 secunde
    while time.time() < timeout and response == "":
        if arduino.in_waiting > 0:
            response = arduino.readline().decode().strip()
        time.sleep(0.1)
    print("Răspuns de la Arduino:", response)


# Bucla principală a jocului de șah
while not board.is_game_over():
    try:
        move = decide_move(board)
        if move is None:
            print("No valid moves left. Game Over.")
            break
        print(f"Best move: {move}")
        servo_index, angle = map_move_to_servos(move)
        send_move_to_arduino(servo_index, angle)
        board.push(move)
        time.sleep(5)  # Așteaptă ca mișcarea să se finalizeze înainte de următoarea comandă
    except Exception as e:
        print(f"Error: {e}")
