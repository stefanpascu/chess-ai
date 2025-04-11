import math
import os
import time

import chess
import numpy as np
import pybullet as p
import serial
from stable_baselines3 import PPO

from reinforcement_learning import settings
from reinforcement_learning.chess_env import ChessEnv  # Mediul tău de șah personalizat
from settings_arm_and_board import board_origin_x, board_margin, square_size, base_height, piece_grabbing_point_length, \
    board_z, board_origin_y, claw_length, upper_arm_length, forearm_length, horizontal_and_vertical_error_margin_length

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

##################### PYBULLET #####################
# p.connect(p.GUI)
# p.setGravity(0, 0, -9.81)
#
# robotStartPos = [0, 0, 0.1]  # Raised slightly to avoid collision with plane
# robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# endEffectorIndex = None
#
# plane_id = p.loadURDF("robot_arm/plane.urdf")
# robotId = p.loadURDF("robot_arm/robot_arm.urdf", robotStartPos, robotStartOrientation, useFixedBase=True)
# def calculate_inverse_kinematics_pybullet(x, y, z, is_grabbing):
#     # Definește poziția țintă ca un vector 3D
#     target_pos = [x, y, z]
#     # Folosește o orientare neutră (poți ajusta Eulerii după necesitate)
#     target_ori = p.getQuaternionFromEuler([0, 0, 0])
#
#     # Calculează valorile articulațiilor folosind kinematicile inverse din PyBullet
#     joint_angles = p.calculateInverseKinematics(robotId, endEffectorIndex, target_pos, target_ori)
#
#     # Convertim valorile obținute din radiani în grade
#     joint_angles_deg = [math.degrees(a) for a in joint_angles]
#
#     # Exemplu de mapping: presupunem că servourile se corespunzător primelor 4 articulații,
#     # iar servo3 și servo5 sunt setate manual, similar cu funcția ta originală.
#     # (Mappingul poate fi ajustat conform configurației robotului tău.)
#     servo0 = joint_angles_deg[0]
#     servo1 = joint_angles_deg[1]
#     servo2 = joint_angles_deg[2]
#     servo4 = joint_angles_deg[3]  # Ajustează indexul în funcție de modelul tău
#     servo3 = 98  # Valoare fixă, similar codului tău
#     servo5 = 0 if is_grabbing else 15
#
#     return [
#         int(round(servo0)),
#         int(round(servo1)),
#         int(round(servo2)),
#         servo3,
#         int(round(servo4)),
#         servo5
#     ]
####################################################


def get_square_center(square):
    """
    Calculează coordonatele centrului unui pătrat de șah.
    'square' este indexul pătratului (0-63) conform python-chess (A1=0, H8=63).
    Returnează (x, y, z) în mm.
    """
    file = chess.square_file(square)  # 0-7 (A-H)
    rank = chess.square_rank(square)  # 0-7 (rândul 1 este 0)
    x = board_origin_x - board_margin - ((file + 0.5) * square_size)
    y = board_origin_y + board_margin + ((8 - (rank + 0.5)) * square_size)
    z = claw_length + board_z + piece_grabbing_point_length - base_height
    # print(x, y, z)
    # print("rank: ", rank)
    return x, y, z


def calculate_inverse_kinematics(x, y, z, is_grabbing):
    print("x, y, z: ", x, y, z)
    b = math.degrees(math.atan2(y, x))
    l = math.sqrt(x * x + y * y) - horizontal_and_vertical_error_margin_length
    h = math.sqrt(l * l + z * z)

    if h <= (upper_arm_length + forearm_length):
        phi = math.degrees(math.atan2(z, l))
        half_h = h / 2.0
        if half_h > upper_arm_length:
            theta = 0
        else:
            theta = math.degrees(math.acos(half_h / upper_arm_length))
        a1 = phi + theta
        a2 = phi - theta
        servo0 = b
        servo1 = a1
        servo2 = (a1 - a2)
        servo4 = -a2
    else:
        z2 = z - claw_length
        h2 = math.sqrt(l * l + z2 * z2)
        if h2 > (upper_arm_length + forearm_length + claw_length):
            return None
        phi2 = math.degrees(math.atan2(z2, l))
        half_h2 = h2 / 2.0
        if half_h2 > upper_arm_length:
            theta2 = 0
        else:
            theta2 = math.degrees(math.acos(half_h2 / upper_arm_length))
        a1_2 = phi2 + theta2
        a2_2 = phi2 - theta2
        servo0 = b
        servo1 = a1_2
        servo2 = (a1_2 - a2_2)
        servo4 = -a2_2

    servo3 = 98
    servo5 = 0 if is_grabbing else 15

    return [
        int(round(servo0)),
        int(round(servo1)),
        int(round(servo2)),
        servo3,
        int(round(servo4)),
        servo5
    ]


def decide_inverse_kinematics_calculation(x, y, z, is_grabbing):
    # input = input("Choose how to calculate inverse kinematics:\n1. PyBullet\n2. Manual")
    input = "2"
    while True:
        if input == "1":
            return
            # return calculate_inverse_kinematics_pybullet(x, y, z, is_grabbing)
        elif input == "2":
            return calculate_inverse_kinematics(x, y, z, is_grabbing)
        else:
            print("Invalid input! \n(type 1 or 2)")

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
    servo_angles_current = decide_inverse_kinematics_calculation(current_x, current_y, current_z, False)
    print(f"Calculated servo angles: {servo_angles_current}")

    target_square = move.to_square
    target_x, target_y, target_z = get_square_center(target_square)
    print(f"Target square center coordinates: x={target_x}, y={target_y}, z={target_z}")
    servo_angles_target = decide_inverse_kinematics_calculation(target_x, target_y, target_z, True)
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
initial_move = chess.Move.from_uci("b2h8")
alternate_move = chess.Move.from_uci("c4d4")
move = initial_move
while not board.is_game_over():
    try:
        move = decide_move(board)
        if move is None:
            print("No valid moves left. Game Over.")
            break
        if move == initial_move:
            move = alternate_move
        else:
            move = initial_move
        print(f"Best move: {move}")
        # send_move_to_arduino([90, 90, 90, 98, 0, 25])
        # time.sleep(1)

        # servo_angles_current, servo_angles_target = map_move_to_robot_arm(move)
        #
        # servo_angles_current[5] = 25
        # send_move_to_arduino(servo_angles_current)
        # time.sleep(1)
        #
        # servo_angles_current[5] = 0
        # send_move_to_arduino(servo_angles_current)
        # time.sleep(1)
        #
        # send_move_to_arduino([90, 90, 90, 98, 0, 5])
        # time.sleep(1)
        #
        # servo_angles_target[5] = 0
        # send_move_to_arduino(servo_angles_target)
        # time.sleep(1)
        #
        # servo_angles_target[5] = 25
        # send_move_to_arduino(servo_angles_target)
        # time.sleep(1)
        #
        # send_move_to_arduino([90, 90, 90, 98, 0, 25])
        # time.sleep(1)


        send_move_to_arduino([90, 90, 90, 98, 90, 25])
        time.sleep(5)
        # servo_angles_current, servo_angles_target = map_move_to_robot_arm(move)
        # send_move_to_arduino(servo_angles_target)
        break


        board.push(move)
    except Exception as e:
        print(f"Error: {e}")