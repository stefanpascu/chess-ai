import math
import os
import time
import chess
import serial
import settings
import numpy as np
import minmax_negamax.negamax as ai_negamax
import reinforcement_learning.rl_settings as rl_settings
from stable_baselines3 import PPO
from reinforcement_learning.chess_env import ChessEnv
from settings import board_origin_x, board_margin, square_size, base_height, piece_grabbing_point_length, \
    board_z, board_origin_y, claw_length, upper_arm_length, forearm_length, horizontal_and_vertical_error_margin_length, \
    weight_error, angle_errors

board = chess.Board()
bits = [1]*16 + [0]*32 + [1]*16
prev_occ = np.array(list(map(int,bits))).reshape(8, 8)

waiting_for_capture = False
capture_sq          = None

arduino = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)

model_path = os.path.join("reinforcement_learning", rl_settings.best_model_path) if os.path.exists(
    os.path.join("reinforcement_learning", rl_settings.best_model_path)
) else os.path.join("reinforcement_learning", rl_settings.reinforced_model_path)
engine = PPO.load(model_path)

env = ChessEnv()
board = chess.Board()


def lay_piece(x, y, z):
    angles = [settings.initial_stance_grabbing]

    servo_angles_target = calculate_inverse_kinematics(x, y, z + settings.lift_piece_height, True, False)
    angles.append(servo_angles_target)

    servo_angles_target[1] += settings.avoid_piece_collision_offset
    angles.append(servo_angles_target)

    servo_angles_target = calculate_inverse_kinematics(x, y, z, True, False)
    angles.append(servo_angles_target)

    servo_angles_target = calculate_inverse_kinematics(x, y, z, False, False)
    angles.append(servo_angles_target)
    if (int(y) != settings.second_line_y_in_cm
            and not (int(y) == settings.third_line_y_in_cm and int(x) == settings.first_column_x_in_cm)
            and not (int(y) == settings.third_line_y_in_cm and int(x) == -settings.first_column_x_in_cm)):
        servo_angles_target = calculate_inverse_kinematics(x, y, z + settings.lift_piece_height,
                                                           False, False)
    else:
        servo_angles_target = calculate_inverse_kinematics(x, y, z,
                                                           False, False)
    angles.append(servo_angles_target)

    angles.append(settings.initial_stance_not_grabbing)

    return angles


def pick_piece(x, y, z):
    angles = [settings.initial_stance_not_grabbing]

    servo_angles_target = calculate_inverse_kinematics(x, y, z + settings.lift_piece_height, False, False)
    angles.append(servo_angles_target)

    servo_angles_target[1] += settings.avoid_piece_collision_offset
    angles.append(servo_angles_target)

    servo_angles_target = calculate_inverse_kinematics(x, y, z, False, False)
    angles.append(servo_angles_target)

    servo_angles_target = calculate_inverse_kinematics(x, y, z, True, False)
    angles.append(servo_angles_target)
    if (int(y) != settings.second_line_y_in_cm
            and not (int(y) == settings.third_line_y_in_cm and int(x) == settings.first_column_x_in_cm)
            and not (int(y) == settings.third_line_y_in_cm and int(x) == -settings.first_column_x_in_cm)):
        servo_angles_target = calculate_inverse_kinematics(x, y, z + settings.lift_piece_height,
                                                                              True, False)
    else:
        servo_angles_target = calculate_inverse_kinematics(x, y, z,
                                                           True, False)
    angles.append(servo_angles_target)
    angles.append(settings.initial_stance_grabbing)

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

    angles.append(settings.captured_piece_dropping_point_closed)
    angles.append(settings.captured_piece_dropping_point_open)
    angles.append(settings.initial_stance_not_grabbing)

    return angles


def map_castling_to_robot_arm(move):
    target_square = move.to_square
    angles = []

    if target_square == chess.G8:
        target_x, target_y, target_z = get_square_center(chess.H8, False)
        angles = pick_piece(target_x, target_y, target_z)
        target_x, target_y, target_z = get_square_center(chess.F8, False)
        angles.extend(lay_piece(target_x, target_y, target_z))
    elif target_square == chess.C8:
        target_x, target_y, target_z = get_square_center(chess.A8, False)
        angles = pick_piece(target_x, target_y, target_z)
        target_x, target_y, target_z = get_square_center(chess.D8, False)
        angles.extend(lay_piece(target_x, target_y, target_z))

    return angles


def get_square_center(square, debug=False):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    x = board_origin_x - board_margin - ((file + 0.5) * square_size)
    y = board_origin_y + board_margin + ((8 - (rank + 0.5)) * square_size)
    z = claw_length + board_z + piece_grabbing_point_length - base_height
    if debug:
        print("############################ CENTRUL PATRATULUI ############################")
        print("file: ", file)
        print("rank: ", rank)
        print("x: {x} \ny: {y} \nz: {z} \n".format(x = x, y = y, z = z))
    return x, y, z


def get_move_from_square_center(x, y, debug=False):
    try :
        file_number = int(round((board_origin_x - board_margin - x) / square_size - 0.5))
        file = chr(ord('a') + file_number)
        rank = int(round((-((y - board_origin_y - board_margin) / square_size - 8) - 0.5) + 1))
        if debug:
            print(f"file_number: {file_number}")
            print(f"file: {file}")
            print(f"rank: {rank}")
            print("file: ", file, "rank: ", rank)
        return str(file) + str(rank)
    except (ValueError, TypeError) as e:
        print("Eroare la transformarea in patrat al tablei:", e)
        return None
    except Exception as e:
        print("Eroare neasteptata:", e)
        return None



def calculate_inverse_kinematics(x, y, z, is_grabbing, debug=False):
    b = math.degrees(math.atan2(y, x))
    l = math.sqrt(x * x + y * y) - horizontal_and_vertical_error_margin_length
    z = z + (l / weight_error)
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
        servo1 = a1
        servo2 = (a1 - a2)
        servo4 = -a2 + weight_error
    else:
        servo1 = 0
        servo2 = 0
        servo4 = 0

    if debug:
        print("b: ", b)
        print("l: ", l)
        print("h: ", h)

    servo0 = b * settings.base_servo_error_exponent
    servo3 = settings.wrist_sideways_angle
    servo5 = settings.claw_closed if is_grabbing else settings.claw_open

    servos = [
        int(round(servo0)),
        int(round(servo1)),
        int(round(servo2)),
        int(round(servo3)),
        int(round(servo4)),
        int(round(servo5))
    ]

    for index in range(6):
        servos[index] = servos[index] + angle_errors[get_move_from_square_center(x, y, debug)][index]

    return servos


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
        return None
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
    servo_angles_current = calculate_inverse_kinematics(current_x, current_y, current_z, False, debug)
    if debug:
        print(f"Coordonatele curente ale patratului: x={current_x}, y={current_y}, z={current_z}")
        print(f"Unghiurile servo calculate: {servo_angles_current}")

    target_square = move.to_square
    target_x, target_y, target_z = get_square_center(target_square, debug)
    servo_angles_target = calculate_inverse_kinematics(target_x, target_y, target_z, True, debug)
    if debug:
        print(f"Coordonatele doreite ale patratului: x={target_x}, y={target_y}, z={target_z}")
        print(f"Unghiurile servo calculate: {servo_angles_target}")

    return servo_angles_current, servo_angles_target


def parse_occupancy(line):
    bits = line.split()
    if len(bits) != 64 or any(b not in ('0','1') for b in bits):
        return None
    vals = list(map(int, bits))
    mat = np.array(vals, dtype=int).reshape((8,8))
    return mat


def detect_move(prev, curr):
    src_idx = np.where((prev==1)&(curr==0))
    dst_idx = np.where((prev==0)&(curr==1))
    if len(src_idx[0])==1 and len(dst_idx[0])==1:
        sr, sc = src_idx[0][0], src_idx[1][0]
        dr, dc = dst_idx[0][0], dst_idx[1][0]
        src_sq = chess.square(sc, sr)
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


def send_move_to_arduino(servo_angles, timeout=15.0, debug=False):
    command = ",".join(str(angle) for angle in servo_angles) + "\n"

    while arduino.in_waiting:
        arduino.read(arduino.in_waiting)

    arduino.write(command.encode())
    arduino.flush()
    if debug:
        print(f"→ Trimitere catre Arduino: {command.strip()}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        if arduino.in_waiting:
            line = arduino.readline().decode('utf-8', 'ignore').strip()
            if debug:
                print("<- Arduino:", repr(line))
            if line == settings.all_servos_done_message:
                print("-> Bratul a terminat mutarea.")
                return
            if line == "MOVE_DONE":
                return
        else:
            time.sleep(0.01)


if __name__ == '__main__':
    action = settings.flow_state
    waiting_for_move_done = False
    move_start_time = 0.0
    move_timeout = 15.0

    if action == "test":
        initial_move = chess.Move.from_uci("b2h8")
        alternate_move = chess.Move.from_uci("a2d4")
        move = initial_move
        while not board.is_game_over():
            try:
                print(board, "\n")
                input_move = input("Introduceti mutarea:\n")
                if input_move[0].isalpha():
                    if input_move == "fin":
                        cm_fen = "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"
                        board.set_fen(cm_fen)
                        if board.is_checkmate():
                            winner = "Negru" if board.turn == chess.WHITE else "Alb"
                            print("Sah mat!", winner, " castiga.")
                            arduino.write(("DEFEAT\n" if board.turn == chess.WHITE else "WIN\n").encode())
                            break
                        if board.is_stalemate():
                            print("Remiza!")
                            arduino.write("STALEMATE\n".encode())
                            break
                        if board.is_insufficient_material():
                            print("Remiza pentru material insuficient.")
                            arduino.write("STALEMATE\n".encode())
                            break
                    elif input_move == "init":
                        send_move_to_arduino([90, 90, 90, 123, 0, 20], move_timeout, False)
                    elif len(input_move) == 4:
                        move = chess.Move.from_uci(input_move)
                        commands = []
                        x, y, z = get_square_center(move.from_square)
                        print(
                            f"from_square_angles: {str(calculate_inverse_kinematics(x, y, z, False, False)).replace(',', '')}")
                        x, y, z = get_square_center(move.to_square)
                        print(
                            f"to_square_angles: {str(calculate_inverse_kinematics(x, y, z, False, False)).replace(',', '')}")
                        commands.extend(map_move_to_angles(move))
                        commands.append(settings.all_angles_sent_message)

                        for seq in commands:
                            if seq == settings.all_angles_sent_message:
                                print(f"seq = {seq}")
                                send_move_to_arduino(seq, move_timeout, False)
                            else:
                                print(f"seq = {' '.join(map(str, seq))}")
                                send_move_to_arduino(seq, move_timeout, False)
                            arduino.flush()
                    elif len(input_move) == 2:
                        commands = []
                        square = chess.parse_square(input_move)
                        x, y, z = get_square_center(square)
                        servo_angles_target = calculate_inverse_kinematics(x, y, z, False, False)
                        commands.append(servo_angles_target)
                        commands.append(settings.all_angles_sent_message)
                        for seq in commands:
                            if seq == settings.all_angles_sent_message:
                                print(f"seq = {seq}")
                                send_move_to_arduino(seq, move_timeout, False)
                            else:
                                print(f"seq = {' '.join(map(str, seq))}")
                                send_move_to_arduino(seq, move_timeout, False)
                            arduino.flush()
                elif input_move[0].isdigit():
                    send_move_to_arduino(input_move.strip().split(" "), move_timeout, False)

            except Exception as e:
                print(f"Eroare: {e}")

    else:
        print("Jocul a inceput")
        debug = True
        while True:
            raw = arduino.readline()
            if not raw:
                continue
            line = raw.decode('utf-8', errors='ignore').strip()
            occ = parse_occ(line)
            if occ is None:
                continue

            src_sqs, dst_sqs = diff_squares(prev_occ, occ)
            board.turn = chess.WHITE
            if not waiting_for_capture:
                if debug:
                    print(f"src_sqs: {src_sqs}")
                    print(f"dst_sqs: {dst_sqs}")
                    print(f"len(src_sqs): {len(src_sqs)}")
                    print(f"len(dst_sqs): {len(dst_sqs)}")
                if len(src_sqs) == 2 and len(dst_sqs) == 2:
                    king_from = next(s for s in src_sqs if board.piece_at(s).piece_type == chess.KING)
                    king_to = next(d for d in dst_sqs if abs(chess.square_file(d) - chess.square_file(king_from)) == 2)
                    mv = chess.Move(king_from, king_to)
                    if mv in board.legal_moves and board.is_castling(mv):
                        board.push(mv)
                        print("Castling move:", mv.uci())
                        if king_to == chess.G1:
                            prev_occ[chess.H1 // 8][chess.H1 % 8] = 0
                            prev_occ[chess.F1 // 8][chess.F1 % 8] = 1
                        elif king_to == chess.C1:
                            prev_occ[chess.A1 // 8][chess.A1 % 8] = 0
                            prev_occ[chess.D1 // 8][chess.D1 % 8] = 1
                        elif king_to == chess.G8:
                            prev_occ[chess.H8 // 8][chess.H8 % 8] = 0
                            prev_occ[chess.F8 // 8][chess.F8 % 8] = 1
                        elif king_to == chess.C8:
                            prev_occ[chess.A8 // 8][chess.A8 % 8] = 0
                            prev_occ[chess.D8 // 8][chess.D8 % 8] = 1
                    else:
                        print("Illegal castling detected:", mv)
                elif len(src_sqs) == 1 and len(dst_sqs) == 1:
                    mv = chess.Move(src_sqs[0], dst_sqs[0])
                    if mv in board.legal_moves:
                        board.push(mv)
                        print("Mutare: ", mv.uci())
                    else:
                        print("Mutare ilegala detectata:", mv)
                elif len(src_sqs) == 1 and len(dst_sqs) == 0:
                    sq = src_sqs[0]
                    piece = board.piece_at(sq)
                    if piece and piece.color != board.turn:
                        waiting_for_capture = True
                        capture_sq = sq
                        arduino.reset_input_buffer()
                        arduino.write(settings.capture_detected_message.encode())
                        arduino.flush()
                        print("Capturare detectata la ", chess.square_name(sq),
                              " — se asteapta mutarea albului ce completeaza capturarea")
                        prev_occ = occ.copy()
            else:
                if len(src_sqs) == 1 and len(dst_sqs) == 1 and dst_sqs[0] == capture_sq:
                    mv = chess.Move(src_sqs[0], dst_sqs[0])
                    if mv in board.legal_moves:
                        board.push(mv)
                        print("Capturare completata:", mv.uci())
                    else:
                        print("Mutare post‐capturare:", mv)
                    waiting_for_capture = False
                    capture_sq = None
                else:
                    print("Jucatirul ar fi trebuit sa mute piesa alba ce captureaza piesa neagra selectata.")

            # AI logic
            print(f"board: \n{board}")
            if not waiting_for_capture:
                if board.is_checkmate():
                    winner = "Negru" if board.turn == chess.WHITE else "Alb"
                    print("Sah mat!", winner, " castiga.")
                    arduino.write(("DEFEAT\n" if board.turn == chess.WHITE else "WIN\n").encode())
                    break
                if board.is_stalemate():
                    print("Remiza!")
                    arduino.write("STALEMATE\n".encode())
                    break
                if board.is_insufficient_material():
                    print("Remiza pentru material insuficient.")
                    arduino.write("STALEMATE\n".encode())
                    break

                prev_occ = occ
                board.turn = chess.BLACK
                move = decide_move(board, settings.ai_type)
                print(f"AI joaca: {move}")
                is_capture = board.is_capture(move)
                is_castling = board.is_castling(move)
                board.push(move)
                prev_occ[move.from_square // 8][move.from_square % 8] = 0
                prev_occ[move.to_square // 8][move.to_square % 8] = 1
                print(f"Tabla: \n{board}")

                commands = []
                if is_castling:
                    removal_seq = map_castling_to_robot_arm(move)
                    commands.extend(removal_seq)
                    if move.to_square == chess.G8:
                        prev_occ[chess.H8 // 8][chess.H8 % 8] = 0
                        prev_occ[chess.F8 // 8][chess.F8 % 8] = 1
                    elif move.to_square == chess.C8:
                        prev_occ[chess.A8 // 8][chess.A8 % 8] = 0
                        prev_occ[chess.D8 // 8][chess.D8 % 8] = 1
                if is_capture:
                    removal_seq = map_capture_removal_to_robot_arm(move)
                    commands.extend(removal_seq)

                commands.extend(map_move_to_angles(move))
                commands.append(settings.all_angles_sent_message)

                for seq in commands:
                    if seq == settings.all_angles_sent_message:
                        print(f"seq = {seq}")
                        send_move_to_arduino(seq, move_timeout, False)
                    else:
                        print(f"seq = {' '.join(map(str, seq))}")
                        send_move_to_arduino(seq, move_timeout, False)
                    arduino.flush()

                if board.is_checkmate():
                    winner = "Negru" if board.turn == chess.WHITE else "Alb"
                    print("Sah mat!", winner, " castiga.")
                    arduino.write(("DEFEAT\n" if board.turn == chess.WHITE else "WIN\n").encode())
                    break
                if board.is_stalemate():
                    print("Remiza!")
                    arduino.write("STALEMATE\n".encode())
                    break
                if board.is_insufficient_material():
                    print("Remiza pentru material insuficient.")
                    arduino.write("STALEMATE\n".encode())
                    break

# TODO: ne asiguram ca toate patratelele sunt "atinse" cum trebuie
# TODO: modalitate a bratului de a arata ca a castigat/pierdut/facut remiza