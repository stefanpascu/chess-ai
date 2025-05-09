# Parametri fizici ai tablei de șah (modificabili)
board_origin_x = 120.0  # Poziția X a colțului din stânga-jos al tablei (mm)
board_origin_y = 50.0  # Poziția Y a colțului din stânga-jos al tablei (mm)
board_margin = 0.0  # Marja de la marginea tablei până la zona de joc (mm)
square_size = 30.0  # Latura fiecărui pătrat (mm)
board_z = 6.0  # Înălțimea suprafeței tablei (mm)

# Parametri ai brațului robotic (ipoteză, conform exemplului HowToMechatronics)
base_height = 100.0  # H: înălțimea bazei (mm)
upper_arm_length = 120.0  # L1: lungimea brațului superior (umăr la cot) (mm)
forearm_length = 123.0  # L2: lungimea brațului inferior (cot la încheietură) (mm)
claw_length = 120.0
forearm_and_claw_error_margin_length = 15.0
horizontal_and_vertical_error_margin_length = 22.6
piece_grabbing_point_length = 10
weight_error = 10

# Unghiurile corespunzatoare fiecarui patrat
angles = {
    "init": [90, 90, 35, 98, 0, 25],
    "a1": [90, 90, 35, 98, 0, 25],
    "a2": [68, 14, 0, 98, 0, 25]
}