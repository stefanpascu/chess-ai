import os
import serial
import time
import chess
import numpy as np
import math
from stable_baselines3 import PPO
from reinforcement_learning import settings
from reinforcement_learning.chess_env import ChessEnv  # Mediul tău de șah personalizat

# Set up Serial communication with Arduino (actualizează portul și baud rate)
arduino = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)  # Așteaptă stabilizarea conexiunii

# Încarcă modelul AI (PPO) de șah
model_path = os.path.join("reinforcement_learning", settings.best_model_path) if os.path.exists(
    os.path.join("reinforcement_learning", settings.best_model_path)
) else os.path.join("reinforcement_learning", settings.reinforced_model_path)
engine = PPO.load(model_path)

# Inițializează mediul de șah și tabla
env = ChessEnv()
board = chess.Board()

# Parametri fizici ai tablei de șah (modificabili)
board_origin_x = 120.0  # Poziția X a colțului din stânga-jos al tablei (mm)
board_origin_y = 318.0  # Poziția Y a colțului din stânga-jos al tablei (mm)
board_margin = 27.0  # Marja de la marginea tablei până la zona de joc (mm)
square_size = 25.0  # Latura fiecărui pătrat (mm)
board_z = 16.0  # Înălțimea suprafeței tablei (mm)

# Parametri ai brațului robotic (ipoteză, conform exemplului HowToMechatronics)
base_height = 95.0  # H: înălțimea bazei (mm)
upper_arm_length = 125.0  # L1: lungimea brațului superior (umăr la cot) (mm)
forearm_length = 125.0  # L2: lungimea brațului inferior (cot la încheietură) (mm)


def get_square_center(square):
    """
    Calculează coordonatele centrului unui pătrat de șah.
    'square' este indexul pătratului (0-63) conform python-chess (A1=0, H8=63).
    Returnează (x, y, z) în mm.
    """
    file = chess.square_file(square)  # 0-7 (A-H)
    rank = chess.square_rank(square)  # 0-7 (rândul 1 este 0)
    x = board_origin_x + board_margin + (file + 0.5) * square_size
    y = board_origin_y + board_margin + (rank + 0.5) * square_size
    # x = board_origin_x - board_margin - (square_size / 2)
    # y = board_origin_y - board_margin - (square_size / 2)
    z = board_z
    return x, y, z


def calculate_inverse_kinematics(x, y, z):
    """
    Calculează un set de 6 unghiuri pentru brațul robotic astfel încât end-effectorul să ajungă la (x, y, z).
    Valorile sunt în milimetri, iar unghiurile rezultate în grade.

    Mapping:
      - Servo 0: rotația bazei (calculată din x, y)
      - Servo 1: umărul (mișcare verticală, calculată din proiecția pe planul vertical)
      - Servo 2: cotul (calculat cu legea cosinusului)
      - Servo 3: încheietura (blocată la 90° pentru a extinde lungimea antebrațului)
      - Servo 4: compensează orientarea wrist-ului (calculat astfel încât efectul total să fie orizontal)
      - Servo 5: cleștele (setat la 90° pentru deschis)
    """
    # 1. Calculul rotației bazei (Servo 0)
    theta0 = math.degrees(math.atan2(y, x))

    # 2. Distanța orizontală de la origine
    R = math.sqrt(x ** 2 + y ** 2)

    # 3. Diferența verticală față de baza brațului (base_height)
    Z = z - base_height

    # 4. Distanța totală de la umăr la țintă
    d = math.sqrt(R ** 2 + Z ** 2)
    if d > (upper_arm_length + forearm_length):
        print("Ținta este inaccesibilă, d =", d)
        d = upper_arm_length + forearm_length  # Saturăm la limita maximă

    # 5. Calculul unghiului de la cot (Servo 2) folosind legea cosinusului:
    cos_angle = (upper_arm_length ** 2 + forearm_length ** 2 - d ** 2) / (2 * upper_arm_length * forearm_length)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    theta_elbow = math.acos(cos_angle)  # în radiani
    servo2 = 180 - math.degrees(theta_elbow)

    # 6. Calculul unghiului umărului (Servo 1)
    cos_shoulder = (upper_arm_length ** 2 + d ** 2 - forearm_length ** 2) / (2 * upper_arm_length * d)
    cos_shoulder = max(-1.0, min(1.0, cos_shoulder))
    theta_shoulder_offset = math.acos(cos_shoulder)  # în radiani
    theta_shoulder_line = math.atan2(Z, R)
    servo1 = math.degrees(theta_shoulder_line + theta_shoulder_offset)

    # 7. Blocăm incheietura la 90° (Servo 3)
    servo3 = 90

    # 8. Calculăm compensarea pentru wrist (Servo 4)
    # Pentru a păstra orientarea orizontală, dorim ca suma efectivă a unghiurilor la umăr, cot și wrist să fie 180°.
    # Dacă servo3 este blocat la 90, atunci servo4 trebuie să fie:
    servo4 = 90 - (servo1 + servo2)

    # 9. Servo 5 (clește) rămâne la 90° (stare deschisă)
    servo5 = 90

    return [int(round(theta0)), int(round(servo1)), int(round(servo2)), servo3, int(round(servo4)), servo5]


def get_observation_from_board(board):
    """
    Convertește starea tablei de șah într-un format (12,8,8) compatibil cu PPO.
    """
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
    """
    Folosește modelul PPO pentru a prezice cea mai bună mișcare de șah.
    """
    observation = get_observation_from_board(board)  # formă (12,8,8)
    observation = np.expand_dims(observation, axis=0)  # formă (1,12,8,8)
    action, _ = engine.predict(observation)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None  # Jocul s-a terminat
    action = int(action.item())
    selected_move = legal_moves[action % len(legal_moves)]
    return selected_move


def map_move_to_robot_arm(move):
    """
    Mapează o mișcare de șah către coordonatele țintă ale brațului robotic,
    calculează inverse kinematics și returnează un set de 6 unghiuri pentru servomotoare.
    Folosește pătratul de destinație al mișcării.
    """
    target_square = move.to_square
    x, y, z = get_square_center(target_square)
    print(f"Target square center coordinates: x={x}, y={y}, z={z}")
    servo_angles = calculate_inverse_kinematics(x, y, z)
    print(f"Calculated servo angles: {servo_angles}")
    return servo_angles


def send_move_to_arduino(servo_angles):
    """
    Trimite comanda completă către Arduino, conținând unghiurile pentru toate cele 6 servomotoare.
    Formatul: "angle0,angle1,angle2,angle3,angle4,angle5\n"
    """
    command = ",".join(str(angle) for angle in servo_angles) + "\n"
    # command = f"{servo_angles[0]},20,60,90,90,90\n"
    arduino.reset_input_buffer()
    arduino.write(command.encode())
    print(f"Sent command to Arduino: {command.strip()}")

    # Așteaptă confirmarea de la Arduino ("MOVE_DONE")
    response = ""
    timeout = time.time() + 10  # Timeout de 10 secunde
    while time.time() < timeout and response == "":
        if arduino.in_waiting > 0:
            response = arduino.readline().decode().strip()
        time.sleep(0.1)
    print("Arduino response:", response)


# Bucla principală a jocului de șah
while not board.is_game_over():
    try:
        move = decide_move(board)
        if move is None:
            print("No valid moves left. Game Over.")
            break
        move = chess.Move.from_uci("a2a1")
        print(f"Best move: {move}")
        servo_angles = map_move_to_robot_arm(move)
        send_move_to_arduino(servo_angles)
        board.push(move)
        time.sleep(5)  # Așteaptă finalizarea mișcării înainte de următoarea comandă
    except Exception as e:
        print(f"Error: {e}")
    break