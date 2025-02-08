# paths
chess_model_path = "models/chess_model.zip"
reinforced_model_path = "models/reinforced_chess_model.zip"
pretrained_model_path = "models/pretrained_chess_model.zip"
pretraining_pgn_data = "pretraining-data/lichess_db_standard_rated_2024-12.pgn/lichess_db_standard_rated_2024-12.pgn"
parsed_pgn_data_with_stockfish = "parsed_chess_dataset_with_stockfish.csv"
parsed_pgn_data = "parsed_chess_dataset.csv"
stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
normalized_env_path = "models/vec_normalize.pkl"
best_model_path = "models/best_model.zip"

# pretraining variables
max_games_for_pretraining = 2**11
epochs_for_pretraining = 5

# training variables
training_number_of_timestamps = 2**22


# will only use for demonstration of optimization and data collection
model_with_pretraining_file_path = "chess_model_with_pretraining.zip"
model_without_pretraining_file_path = "chess_model_without_pretraining.zip"