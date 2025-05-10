import math
import os
import time

import chess
import numpy as np
import pybullet as p
import serial
from stable_baselines3 import PPO

import minmax_negamax.negamax as ai_negamax

import reinforcement_learning.settings as settings
import settings_measurements
from reinforcement_learning.chess_env import ChessEnv  # Mediul tău de șah personalizat
from settings_measurements import board_origin_x, board_margin, square_size, base_height, piece_grabbing_point_length, \
    board_z, board_origin_y, claw_length, upper_arm_length, forearm_length, horizontal_and_vertical_error_margin_length, \
    forearm_and_claw_error_margin_length, weight_error

INITIAL_STANCE_GRABBING = [90, 90 ,90 , 123, 0, 0]
INITIAL_STANCE_NOT_GRABBING = [90, 90 ,90 , 123, 0, 25]

board     = chess.Board()
prev_occ  = chess.Board()
waiting_for_capture = False
capture_sq          = None

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
def setup_pybullet():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    robotStartPos = [0, 0, 0.1]  # Raised slightly to avoid collision with plane
    robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    global end_effector_index
    end_effector_index = None

    plane_id = p.loadURDF("robot_arm/plane.urdf")
    global robot_id
    robot_id = p.loadURDF("robot_arm/robot_arm.urdf", robotStartPos, robotStartOrientation, useFixedBase=True)


def calculate_inverse_kinematics_pybullet(x, y, z, is_grabbing):
    setup_pybullet()
    # Definește poziția țintă ca un vector 3D
    target_pos = [x, y, z]
    # Folosește o orientare neutră (poți ajusta Eulerii după necesitate)
    target_ori = p.getQuaternionFromEuler([0, 0, 0])

    # Calculează valorile articulațiilor folosind kinematicile inverse din PyBullet
    joint_angles = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos, target_ori)

    # Convertim valorile obținute din radiani în grade
    joint_angles_deg = [math.degrees(a) for a in joint_angles]

    # Exemplu de mapping: presupunem că servourile se corespunzător primelor 4 articulații,
    # iar servo3 și servo5 sunt setate manual, similar cu funcția ta originală.
    # (Mappingul poate fi ajustat conform configurației robotului tău.)
    servo0 = joint_angles_deg[0]
    servo1 = joint_angles_deg[1]
    servo2 = joint_angles_deg[2]
    servo4 = joint_angles_deg[3]  # Ajustează indexul în funcție de modelul tău
    servo3 = 98  # Valoare fixă, similar codului tău
    servo5 = 0 if is_grabbing else 15

    return [
        int(round(servo0)),
        int(round(servo1)),
        int(round(servo2)),
        servo3,
        int(round(servo4)),
        servo5
    ]
####################################################


def lay_piece(x, y, z):
    angles = [INITIAL_STANCE_GRABBING]

    servo_angles_target = decide_inverse_kinematics_calculation(x, y, z + settings_measurements.lift_piece_height, True, False)
    angles.append(servo_angles_target)

    servo_angles_target = decide_inverse_kinematics_calculation(x, y, z, True, False)
    angles.append(servo_angles_target)

    servo_angles_target = decide_inverse_kinematics_calculation(x, y, z, False, False)
    angles.append(servo_angles_target)

    servo_angles_target = decide_inverse_kinematics_calculation(x, y, z + settings_measurements.lift_piece_height,
                                                                              False, False)
    angles.append(servo_angles_target)

    angles.append(INITIAL_STANCE_NOT_GRABBING)

    return angles


def pick_piece(x, y, z):
    angles = [INITIAL_STANCE_NOT_GRABBING]

    servo_angles_target = decide_inverse_kinematics_calculation(x, y, z + settings_measurements.lift_piece_height, False, False)
    angles.append(servo_angles_target)

    servo_angles_target = decide_inverse_kinematics_calculation(x, y, z, False, False)
    angles.append(servo_angles_target)

    servo_angles_target = decide_inverse_kinematics_calculation(x, y, z, True, False)
    angles.append(servo_angles_target)

    servo_angles_target = decide_inverse_kinematics_calculation(x, y, z + settings_measurements.lift_piece_height,
                                                                              True, False)
    angles.append(servo_angles_target)

    angles.append(INITIAL_STANCE_GRABBING)

    return angles


def map_move_to_angles(move):
    current_square = move.from_square
    current_x, current_y, current_z = get_square_center(current_square, False)

    angles = pick_piece(current_x, current_y, current_z)

    target_square = move.to_square
    target_x, target_y, target_z = get_square_center(target_square, False)

    angles.extend(lay_piece(target_x, target_y, target_z))

    return angles


def map_capture_removal_to_robot_arm(move):
    target_square = move.to_square
    target_x, target_y, target_z = get_square_center(target_square, False)

    angles = pick_piece(target_x, target_y, target_z)

    angles.append([0, 90, 90, 123, 0, 0])
    angles.append([0, 90, 90, 123, 0, 35])
    angles.append(INITIAL_STANCE_NOT_GRABBING)

    return angles


def get_square_center(square, debug=False):
    file = chess.square_file(square)  # 0-7 (A-H)
    rank = chess.square_rank(square)  # 0-7 (rândul 1 este 0)
    x = board_origin_x - board_margin - ((file + 0.5) * square_size)
    y = board_origin_y + board_margin + ((8 - (rank + 0.5)) * square_size)
    z = claw_length + board_z + piece_grabbing_point_length - base_height
    if debug:
        print("############################ GET SQUARE CENTER ############################")
        print("file: ", file)
        print("rank: ", rank)
        print("x: {x} \ny: {y} \nz: {z} \n".format(x = x, y = y, z = z))
    return x, y, z


def get_move_from_square_center(x, y, z):
    file = (board_origin_x - board_margin - x) / square_size - 0.5
    rank = -((y - board_origin_y - board_margin) / square_size - 8) - 0.5
    print("file: ", file, "rank: ", rank)
    return file, rank


def calculate_inverse_kinematics(x, y, z, is_grabbing, debug=False):
    b = math.degrees(math.atan2(y, x))
    l = math.sqrt(x * x + y * y) - horizontal_and_vertical_error_margin_length
    z = z + (l / 10) # offset for the weight of the arm
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

        if debug:
            print("{h} <= {arm}".format(h=h, arm=upper_arm_length + forearm_length))
            print("phi: ", phi)
            print("half_h: ", half_h)
            print("theta: ", theta)
            print("a1: ", a1)
            print("a2: ", a2)
        # servo1 = a1 + 8
        servo1 = a1
        servo2 = (a1 - a2)
        servo4 = -a2 + weight_error
        # servo4 = -a2 + 35
    else:
        # TODO use hardcoded angles
        print("################### HARDCODED ANGLES NOT WORKING YET ###################")

    if debug:
        print("############################ MANUALLY CALCULATING INVERSE KINEMATICS ############################")
        print("b: ", b)
        print("l: ", l)
        print("h: ", h)

    servo0 = b
    servo3 = 123
    servo5 = 0 if is_grabbing else 25

    return [
        int(round(servo0)),
        int(round(servo1)),
        int(round(servo2)),
        servo3,
        int(round(servo4)),
        servo5
    ]


def decide_inverse_kinematics_calculation(x, y, z, is_grabbing, debug=False):
    calculation_type = "MANUAL"
    while True:
        if calculation_type == "PYBULLET":
            return calculate_inverse_kinematics_pybullet(x, y, z, is_grabbing)
        elif calculation_type == "MANUAL":
            return calculate_inverse_kinematics(x, y, z, is_grabbing, debug)
        elif calculation_type == "HARDCODED":
            file, rank = get_move_from_square_center(x, y, z)
            aux = chess.square_name(chess.square(int(file), int(rank)))
            return settings_measurements.angles[str(aux)]
        else:
            print("Invalid calculation type for inverse kinematics")


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


def decide_move(board, ai_type):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None # game is over
    selected_move = None
    if ai_type == "RL":
        observation = get_observation_from_board(board)  # (12,8,8)
        observation = np.expand_dims(observation, axis=0)  # (1,12,8,8)
        action, _ = engine.predict(observation)
        action = int(action.item())
        selected_move = legal_moves[action % len(legal_moves)]
    if ai_type == "NM":
        selected_move = ai_negamax.find_best_move(board, legal_moves)
        if selected_move is None:
            selected_move = ai_negamax.find_random_move(legal_moves)
    if selected_move is not None:
        return selected_move
    raise Exception("Error: ", "Move is None")


def map_move_to_robot_arm(move, debug=False):
    current_square = move.from_square
    current_x, current_y, current_z = get_square_center(current_square, debug)
    servo_angles_current = decide_inverse_kinematics_calculation(current_x, current_y, current_z, False, debug)
    if debug:
        print(f"Current square center coordinates: x={current_x}, y={current_y}, z={current_z}")
        print(f"Calculated servo angles: {servo_angles_current}")

    target_square = move.to_square
    target_x, target_y, target_z = get_square_center(target_square, debug)
    servo_angles_target = decide_inverse_kinematics_calculation(target_x, target_y, target_z, True, debug)
    if debug:
        print(f"Target square center coordinates: x={target_x}, y={target_y}, z={target_z}")
        print(f"Calculated servo angles: {servo_angles_target}")

    return servo_angles_current, servo_angles_target


def send_move_to_arduino(servo_angles):
    command = ",".join(str(angle) for angle in servo_angles) + "\n"
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
    arduino.flush()


def parse_occupancy(line):
    bits = line.split()
    if len(bits) != 64 or any(b not in ('0','1') for b in bits):
        return None
    # turn into ints
    vals = list(map(int, bits))
    # reshape into 8×8, assuming sensors send a1…h1,a2…h2,…,a8…h8:
    mat = np.array(vals, dtype=int).reshape((8,8))
    return mat


def detect_move(prev, curr):
    # source is where prev==1 and curr==0
    src_idx = np.where((prev==1)&(curr==0))
    dst_idx = np.where((prev==0)&(curr==1))
    if len(src_idx[0])==1 and len(dst_idx[0])==1:
        # convert (row, col) to square index
        # row 0 → rank 1, row 7 → rank 8; col 0 → file a
        sr, sc = src_idx[0][0], src_idx[1][0]
        dr, dc = dst_idx[0][0], dst_idx[1][0]
        src_sq = chess.square(sc, sr)  # sc=file, sr=rank
        dst_sq = chess.square(dc, dr)
        return chess.Move(src_sq, dst_sq)
    return None


def parse_occ(line):
    bits = line.split()
    if len(bits)!=64 or any(b not in ('0','1') for b in bits):
        return None
    arr = np.array(list(map(int,bits))).reshape((8,8))
    return arr

def diff_squares(prev, curr):
    src = np.where((prev==1)&(curr==0))
    dst = np.where((prev==0)&(curr==1))
    src_sqs = [chess.square(c, r) for r,c in zip(*src)]
    dst_sqs = [chess.square(c, r) for r,c in zip(*dst)]
    return src_sqs, dst_sqs


def wait_for_move_done(timeout=15.0):
    deadline = time.time() + timeout
    while True:
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for MOVE_DONE")
        raw = arduino.readline()
        if not raw:
            continue
        if raw.decode('utf-8', errors='ignore').strip() == "MOVE_DONE":
            return


if __name__ == '__main__':
    action = "run"
    if action == "test":
        initial_move = chess.Move.from_uci("b2h8")
        alternate_move = chess.Move.from_uci("a2d4")
        move = initial_move
        while not board.is_game_over():
            try:
                print(board, "\n")
                if move is None:
                    print("No valid moves left. Game Over.")
                    break
                if move == initial_move:
                    move = alternate_move
                else:
                    move = initial_move
                print(f"Best move: {move}")
                # initial position: 90 90 90 123 0 25
                # some    position: 70 70 30 98 60 40
                # A1: 71 12 0 98 42 25
                # A2: 70 14 0 98 28 25
                # A3: 65 40 50 98 49 25
                # A4: 62 50 68 98 54 25
                # A5: 57 62 88 98 65 25
                # A6: 53 70 101 98 71 25

                input_move = input("Input move:\n")
                if input_move == "init":
                    send_move_to_arduino([90, 90, 90, 123, 0, 25])
                elif input_move == "custom":
                    send_move_to_arduino(input("Input custom angles:\n").split(" "))
                else:
                    move = chess.Move.from_uci(input_move)
                    servo_angles_current, servo_angles_target = map_move_to_robot_arm(move, True)
                    send_move_to_arduino(servo_angles_current)

            except Exception as e:
                print(f"Error: {e}")

    else:
        print("Starting…")

        while True:
            raw = arduino.readline()
            if not raw:
                continue
            line = raw.decode('utf-8', errors='ignore').strip()
            occ = parse_occ(line)
            if occ is None:
                continue

            print(f"Occupation matrix: \n{occ}")
            if prev_occ is None:
                prev_occ = occ
                continue

            src_sqs, dst_sqs = diff_squares(prev_occ, occ)

            board.turn = chess.WHITE
            if not waiting_for_capture:
                # 1) Normal one‐to‐one move
                if len(src_sqs) == 1 and len(dst_sqs) == 1:
                    mv = chess.Move(src_sqs[0], dst_sqs[0])
                    if mv in board.legal_moves:
                        board.push(mv)
                        print("Move:", mv.uci())
                    else:
                        print("Illegal move detected:", mv)
                # 2) Possible capture: piece vanished but no arrival
                elif len(src_sqs) == 1 and len(dst_sqs) == 0:
                    sq = src_sqs[0]
                    piece = board.piece_at(sq)
                    # only treat as capture if that piece was of the side to be captured
                    if piece and piece.color != board.turn:
                        waiting_for_capture = True
                        capture_sq = sq
                        print("Capture detected at", chess.square_name(sq),
                              "— waiting for the white arrival move")
                # else: noise or multi‐move; ignore
            else:
                # we were waiting for the capture‐finishing move
                # look for exactly one src+dst, and dst must be capture_sq
                if len(src_sqs) == 1 and len(dst_sqs) == 1 and dst_sqs[0] == capture_sq:
                    mv = chess.Move(src_sqs[0], dst_sqs[0])
                    if mv in board.legal_moves:
                        board.push(mv)
                        print("Capture completed:", mv.uci())
                    else:
                        print("Illegal post‐capture move:", mv)
                    waiting_for_capture = False
                    capture_sq = None

            # after any push, check for game end
            if board.is_checkmate():
                winner = "Black" if board.turn == chess.WHITE else "White"
                print("Checkmate!", winner, "wins.")
                break
            if board.is_stalemate():
                print("Stalemate!")
                break
            if board.is_insufficient_material():
                print("Draw by insufficient material.")
                break

            prev_occ = occ

            # AI response
            board.turn = chess.BLACK
            move = decide_move(board, "NM")
            print(f"AI plays: {move}")
            is_capture = board.is_capture(move)
            board.push(move)

            commands = []
            if is_capture:
                removal_seq = map_capture_removal_to_robot_arm(move)
                commands.extend(removal_seq)

            commands.extend(map_move_to_angles(move))
            commands.append("ALL_ANGLES_SENT")

            arduino.reset_input_buffer()

            for seq in commands:
                print(f"angles: {seq}")
                # send the next waypoint
                send_move_to_arduino(seq)

                # now block until the Arduino acknowledges completion
                try:
                    wait_for_move_done(timeout=30.0)
                except TimeoutError as e:
                    print("Warning:", e)