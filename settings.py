
################################## SETARI AJUSTABILE ##################################
ai_type                 = "NM"  # NM - negamax
                                # RL - reinforcement learning
minmax_negamax_depth    = 5     # Valoarea trebuie sa nu depaseasca valoarea 5
flow_state              = "run" # test - pentru debugging
                                # run  - mod normal de joc


############################### MASURI (A NU SE SCHIMBA) ##############################

# mesaje Arduino
all_servos_done_message     = "ALL_SERVOS_DONE"
all_angles_sent_message     = "ALL_ANGLES_SENT"
capture_detected_message    = "CAPTURE_DETECTED\n"

# Parametri pentru unghiurile servomotoarelor
initial_stance_grabbing                 = [90, 90 ,90 , 123, 0, 0]
initial_stance_not_grabbing             = [90, 90 ,90 , 123, 0, 20]
captured_piece_dropping_point_closed    = [0, 90, 90, 123, 0, 0]
captured_piece_dropping_point_open      = [0, 90, 90, 123, 0, 35]
base_servo_error_exponent   = 0.95
wrist_sideways_angle        = 123
claw_open                   = 20
claw_closed                 = 0
piece_grabbing_point_length = 20
weight_error                = 10
second_line_y_in_cm         = 245

# Parametri fizici ai tablei de șah (modificabili)
board_origin_x                  = 120.0  # Poziția X a colțului din stânga-jos al tablei (mm)
board_origin_y                  = 50.0  # Poziția Y a colțului din stânga-jos al tablei (mm)
board_margin                    = 0.0  # Marja de la marginea tablei până la zona de joc (mm)
square_size                     = 30.0  # Latura fiecărui pătrat (mm)
board_z                         = 6.0  # Înălțimea suprafeței tablei (mm)
lift_piece_height               = 45 # Inaltimea la care trebuie sa ajunga piesa pentru a nu darama alte piese
avoid_piece_collision_offset    = 30

# Parametri ai brațului robotic
base_height                                 = 100.0  # H: înălțimea bazei (mm)
upper_arm_length                            = 120.0  # L1: lungimea brațului superior (umăr la cot) (mm)
forearm_length                              = 123.0  # L2: lungimea brațului inferior (cot la încheietură) (mm)
claw_length                                 = 120.0
forearm_and_claw_error_margin_length        = 15.0
horizontal_and_vertical_error_margin_length = 22.6

# Erorile unghiurilor corespunzatoare fiecarui patrat
angle_errors = {
    "init": [90, 90, 90, 123, 0, 20],
    "a1": [3, 15, 0, 0, 9, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "a2": [3, 23, 12, 0, 2, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "a3": [2, 0, 0, -1, -3, 0],
    "a4": [2, 0, 0, 0, -2, 0],
    "a5": [1, 0, 0, 0, -3, 0],
    "a6": [2, -2, 0, 0, -3, 0],
    "a7": [2, 0, 0, 0, -4, 0],
    "a8": [3, 0, 0, 0, -2, 0],
    "b1": [4, 15, 0, 0, 5, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "b2": [2, 30, 25, 0, 8, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "b3": [1, -1, 0, 0, -2, 0],
    "b4": [0, 0, 0, 0, -2, 0],
    "b5": [0, -3, 0, 0, -3, 0],
    "b6": [0, -4, 0, 0, -1, 0],
    "b7": [0, -2, 0, 0, 0, 0],
    "b8": [3, -2, 0, 0, 0, 0],
    "c1": [4, 15, 0, 0, 3, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "c2": [3, 0, 0, 0, -3, 0],
    "c3": [3, -1, 0, 0, -2, 0],
    "c4": [0, -2, 0, 0, 0, 0],
    "c5": [0, -2, 0, 0, -2, 0],
    "c6": [0, -3, 0, 0, 0, 0],
    "c7": [0, -4, 0, 0, 0, 0],
    "c8": [-5, -3, 0, 0, 1, 0],
    "d1": [4, 17, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "d2": [3, -1, 0, 0, -3, 0],
    "d3": [2, -2, 0, 0, 0, 0],
    "d4": [2, 2, 0, 0, 1, 0],
    "d5": [1, -3, 0, 0, -1, 0],
    "d6": [0, -4, 0, 0, 0, 0],
    "d7": [-1, -4, 0, 0, 0, 0],
    "d8": [-7, -4, 0, 0, 0, 0],
    "e1": [6, 17, 0, 0, 0, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "e2": [4, -1, 0, 0, -1, 0],
    "e3": [4, -1, 0, 0, -1, 0],
    "e4": [3, -2, 0, 0, 1, 0],
    "e5": [2, -2, 0, 0, 0, 0],
    "e6": [1, -2, 0, 0, 0, 0],
    "e7": [0, -2, 0, 0, 0, 0],
    "e8": [-9, -3, 0, 0, -1, 0],
    "f1": [6, 16, 0, 0, 1, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "f2": [5, 0, 0, 0, -3, 0],
    "f3": [4, -1, 0, 0, -3, 0],
    "f4": [4, -1, 0, 0, -1, 0],
    "f5": [2, -2, 0, 0, -2, 0],
    "f6": [2, -2, 0, 0, -2, 0],
    "f7": [-1, -2, 0, 0, 0, 0],
    "f8": [-9, 0, 0, 0, 0, 0],
    "g1": [6, 15, 0, 0, 5, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "g2": [5, 30, 27, 0, 9, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "g3": [3, 0, 0, 0, -2, 0],
    "g4": [4, -2, 0, 0, -3, 0],
    "g5": [3, -3, 0, 0, -3, 0],
    "g6": [1, -3, 0, 0, -2, 0],
    "g7": [1, -1, 0, 0, -1, 0],
    "g8": [-2, -1, 0, 0, -2, 0],
    "h1": [5, 15, 1, 0, 12, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "h2": [5, 25, 17, 0, 6, 0], # acest patrat nu poaet fi atins cu modul de calcul actual
    "h3": [5, 0, 0, 0, -4, 0],
    "h4": [4, -1, 0, 0, -3, 0],
    "h5": [3, -2, 0, 0, -1, 0],
    "h6": [2, -3, 0, 0, -2, 0],
    "h7": [1, -2, 0, 0, -4, 0],
    "h8": [1, -1, 0, 0, -3, 0]
}
