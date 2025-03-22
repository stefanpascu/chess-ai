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
board_origin_y = 70.0  # Poziția Y a colțului din stânga-jos al tablei (mm)
board_margin = 20.0  # Marja de la marginea tablei până la zona de joc (mm)
square_size = 24.0  # Latura fiecărui pătrat (mm)
board_z = 15.0  # Înălțimea suprafeței tablei (mm)

# Parametri ai brațului robotic (ipoteză, conform exemplului HowToMechatronics)
base_height = 100.0  # H: înălțimea bazei (mm)
upper_arm_length = 120.0  # L1: lungimea brațului superior (umăr la cot) (mm)
forearm_length = 120.0  # L2: lungimea brațului inferior (cot la încheietură) (mm)
claw_height = 130.0
forearm_and_claw_error_margin_length = 15.0
piece_grabbing_point_length = 2

def get_square_center(square):
    """
    Calculează coordonatele centrului unui pătrat de șah.
    'square' este indexul pătratului (0-63) conform python-chess (A1=0, H8=63).
    Returnează (x, y, z) în mm.
    """
    file = chess.square_file(square)  # 0-7 (A-H)
    rank = chess.square_rank(square)  # 0-7 (rândul 1 este 0)
    x = board_origin_x - board_margin - (file + 0.5) * square_size
    y = board_origin_y + board_margin + (7 - (rank + 0.5)) * square_size
    z = claw_height + board_z + piece_grabbing_point_length - base_height
    return x, y, z


def calculate_inverse_kinematics(x, y, z, is_grabbing):
    b = math.degrees(math.atan2(y, x)) # * (180 / math.pi)

    # base_offset = 17.0  # example offset from the center of rotation
    # l = math.sqrt(x*x + y*y) - base_offset
    l = math.sqrt(x*x + y*y)

    # h = math.sqrt(l**2 + z**2) - forearm_and_claw_error_margin_length
    h = math.sqrt(l*l + z*z)

    # phi = math.atan(z / l) * (180 / math.pi)
    phi = math.degrees(math.atan2(z, l))

    # theta = math.acos((h / 2) / 120) * (180 / math.pi)
    link_length = 120.0
    half_h = h / 2.0
    if half_h > link_length:
        # Out of reach, or you could clamp the value
        # to avoid math domain error in acos
        theta = 0
    else:
        theta = math.degrees(math.acos(half_h / link_length))

    a1 = phi + theta
    a2 = phi - theta

    servo0 = b
    servo1 = a1
    servo2 = 0 + (a1 - a2)
    servo3 = 85

    # -- 5) Wrist angle (servo4) to keep end-effector vertical (down) --
    # In a simple 2-link planar arm, the final orientation is (shoulder + elbow).
    # We want that final orientation to be 90° in the plane if "90°" means "straight down."
    #
    # totalOrientation = servo1 + (servo2 - 180)
    # We want the end-effector to remain at 90°, so:
    #   servo4 = 90 - totalOrientation
    # Substituting servo2 = a2 + 180 => totalOrientation = servo1 + a2
    # but in code, "servo2" is already (a2 + 180). So:
    #   servo4 = 90 - (servo1 + (servo2 - 180)) = 270 - servo1 - servo2
    #
    # That keeps the wrist pointing "down" in the vertical plane.
    servo4 = 0 - a2

    # servo4 = 90
    if is_grabbing:
        servo5 = 15 # closed
    else:
        servo5 = 25 # open

    return [int(round(servo0)), int(round(servo1)), int(round(servo2)), servo3, int(round(servo4)), servo5]
    # return [90, 90, 90, 85, 0, 25]


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
    current_square = move.from_square
    current_x, current_y, current_z = get_square_center(current_square)
    print(f"Current square center coordinates: x={current_x}, y={current_y}, z={current_z}")
    servo_angles_current = calculate_inverse_kinematics(current_x, current_y, current_z, False)
    print(f"Calculated servo angles: {servo_angles_current}")

    target_square = move.to_square
    target_x, target_y, target_z = get_square_center(target_square)
    print(f"Target square center coordinates: x={target_x}, y={target_y}, z={target_z}")
    servo_angles_target = calculate_inverse_kinematics(target_x, target_y, target_z, True)
    print(f"Calculated servo angles: {servo_angles_target}")

    return servo_angles_current, servo_angles_target


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
initial_move = chess.Move.from_uci("d3a3")
alternate_move = chess.Move.from_uci("a3d3")
move = initial_move
while not board.is_game_over():
    try:
        # move = decide_move(board)
        # if move is None:
        #     print("No valid moves left. Game Over.")
        #     break
        if move == initial_move:
            move = alternate_move
        else:
            move = initial_move
        print(f"Best move: {move}")
        send_move_to_arduino([90, 90, 90, 85, 0, 30])
        time.sleep(5)

        servo_angles_current, servo_angles_target = map_move_to_robot_arm(move)

        servo_angles_current[5] = 30
        send_move_to_arduino(servo_angles_current)
        time.sleep(5)

        servo_angles_current[5] = 5
        send_move_to_arduino(servo_angles_current)
        time.sleep(5)

        send_move_to_arduino([90, 90, 90, 85, 0, 5])
        time.sleep(5)

        servo_angles_target[5] = 5
        send_move_to_arduino(servo_angles_target)
        time.sleep(5)

        servo_angles_target[5] = 30
        send_move_to_arduino(servo_angles_target)
        time.sleep(5)

        send_move_to_arduino([90, 90, 90, 85, 0, 30])
        time.sleep(5)
        board.push(move)
    except Exception as e:
        print(f"Error: {e}")
    # break