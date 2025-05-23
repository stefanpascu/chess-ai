# Parametri fizici ai tablei de șah (modificabili)
board_origin_x = 120.0  # Poziția X a colțului din stânga-jos al tablei (mm)
board_origin_y = 50.0  # Poziția Y a colțului din stânga-jos al tablei (mm)
board_margin = 0.0  # Marja de la marginea tablei până la zona de joc (mm)
square_size = 30.0  # Latura fiecărui pătrat (mm)
board_z = 6.0  # Înălțimea suprafeței tablei (mm)
lift_piece_height = 45 # Inaltimea la care trebuie sa ajunga piesa pentru a nu darama alte piese
avoid_piece_collision_offset = 30

# Parametri ai brațului robotic (ipoteză, conform exemplului HowToMechatronics)
base_height = 100.0  # H: înălțimea bazei (mm)
upper_arm_length = 120.0  # L1: lungimea brațului superior (umăr la cot) (mm)
forearm_length = 123.0  # L2: lungimea brațului inferior (cot la încheietură) (mm)
claw_length = 120.0
forearm_and_claw_error_margin_length = 15.0
horizontal_and_vertical_error_margin_length = 22.6
piece_grabbing_point_length = 20
weight_error = 10

# Unghiurile corespunzatoare fiecarui patrat
angle_errors = {
    "init": [90, 90, 90, 123, 0, 20],
    "a1": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "a2": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "a3": [2, 0, 0, -1, -3, 0], # done
    "a4": [2, 0, 0, 0, -2, 0], # done
    "a5": [1, 0, 0, 0, -3, 0], # done
    "a6": [2, -2, 0, 0, -3, 0], # done
    "a7": [2, 0, 0, 0, -4, 0], # done
    "a8": [3, 0, 0, 0, -2, 0], # done
    "b1": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "b2": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "b3": [1, -1, 0, 0, -2, 0], # done
    "b4": [0, 0, 0, 0, -2, 0], # done
    "b5": [0, -3, 0, 0, -3, 0], # done
    "b6": [0, -4, 0, 0, -1, 0], # done
    "b7": [0, -2, 0, 0, 0, 0], # done
    "b8": [3, -2, 0, 0, 0, 0], # done
    "c1": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "c2": [3, 0, 0, 0, -3, 0], # done
    "c3": [3, -1, 0, 0, -2, 0], # done
    "c4": [0, -2, 0, 0, 0, 0], # done
    "c5": [0, -2, 0, 0, -2, 0], # done
    "c6": [0, -3, 0, 0, 0, 0], # done
    "c7": [0, -4, 0, 0, 0, 0], # done
    "c8": [-5, -3, 0, 0, 1, 0], # done
    "d1": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "d2": [3, -1, 0, 0, -3, 0], # done
    "d3": [2, -2, 0, 0, 0, 0], # done
    "d4": [2, 2, 0, 0, 1, 0], # done
    "d5": [1, -3, 0, 0, -1, 0], # done
    "d6": [0, -4, 0, 0, 0, 0], # done
    "d7": [-1, -4, 0, 0, 0, 0], # done
    "d8": [-7, -4, 0, 0, 0, 0], # done
    "e1": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "e2": [4, -1, 0, 0, -1, 0], # done
    "e3": [4, -1, 0, 0, -1, 0], # done
    "e4": [3, -2, 0, 0, 1, 0], # done
    "e5": [2, -2, 0, 0, 0, 0], # done
    "e6": [1, -2, 0, 0, 0, 0], # done
    "e7": [0, -2, 0, 0, 0, 0], # done
    "e8": [-9, -3, 0, 0, -1, 0], # done
    "f1": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "f2": [0, 0, 0, 0, 0, 0],
    "f3": [0, 0, 0, 0, 0, 0],
    "f4": [0, 0, 0, 0, 0, 0],
    "f5": [0, 0, 0, 0, 0, 0],
    "f6": [0, 0, 0, 0, 0, 0],
    "f7": [0, 0, 0, 0, 0, 0],
    "f8": [0, 0, 0, 0, 0, 0],
    "g1": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "g2": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "g3": [0, 0, 0, 0, 0, 0],
    "g4": [0, 0, 0, 0, 0, 0],
    "g5": [0, 0, 0, 0, 0, 0],
    "g6": [0, 0, 0, 0, 0, 0],
    "g7": [0, 0, 0, 0, 0, 0],
    "g8": [0, 0, 0, 0, 0, 0],
    "h1": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "h2": [0, 0, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "h3": [0, 0, 0, 0, 0, 0],
    "h4": [0, 0, 0, 0, 0, 0],
    "h5": [0, 0, 0, 0, 0, 0],
    "h6": [0, 0, 0, 0, 0, 0],
    "h7": [0, 0, 0, 0, 0, 0],
    "h8": [0, 0, 0, 0, 0, 0]
}