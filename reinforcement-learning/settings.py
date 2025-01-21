# paths
chess_model_path = "chess_model.zip"
reinforced_model_path = "reinforced_chess_model.zip"
pretrained_model_path = "pretrained_chess_model.zip"
pretraining_pgn_data = "pretraining-data/lichess_elite_2021-11/lichess_elite_2021-11.pgn"
parsed_pgn_data = "parsed_chess_dataset_with_stockfish.csv"
stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
normalized_env_path = "logs/vec_normalize.pkl"

# pretraining variables
max_games_for_pretraining = 1000

# training variables
training_number_of_timestamps = 10000


# will only use for demonstration of optimization and data collection
model_with_pretraining_file_path = "chess_model_with_pretraining.zip"
model_without_pretraining_file_path = "chess_model_without_pretraining.zip"