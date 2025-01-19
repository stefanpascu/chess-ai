import chess
import numpy as np
from gymnasium import Env
from gymnasium import spaces
from stockfish import Stockfish  # Assuming Stockfish is installed and available for move quality

# Initialize Stockfish (optional for evaluating move quality)
stockfish = Stockfish(path="stockfish/stockfish-windows-x86-64-avx2.exe")


class ChessEnv(Env):
    def __init__(self, render_mode=None):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.previous_board = None

        # Define the action space and observation space
        self.action_space = spaces.Discrete(4672)  # Adjust for all possible chess moves
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12, 8, 8), dtype=np.float64)

        # Store the render mode (e.g., 'human' for console printing, or others if needed)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state and returns an initial observation."""
        super().reset(seed=seed)
        self.board.reset()
        self.previous_board = None
        return self.get_observation(), {}

    def step(self, action):
        """Executes a move and returns the observation, reward, done, truncated, and info."""
        legal_moves = list(self.board.legal_moves)
        move = legal_moves[action % len(legal_moves)]
        self.board.push(move)

        done = self.board.is_game_over()
        reward = self.get_reward()

        # Set truncated to False as it doesn't apply in chess
        truncated = False

        return self.get_observation(), reward, done, truncated, {}

    def get_observation(self):
        """
        Converts the chess board state into a (12, 8, 8) numerical tensor.
        Each layer of the tensor represents a specific piece type:
        0-5: White pieces (Pawns, Knights, Bishops, Rooks, Queens, Kings)
        6-11: Black pieces (Pawns, Knights, Bishops, Rooks, Queens, Kings)
        """
        obs = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                row, col = divmod(i, 8)
                layer = piece_map[piece.piece_type]
                if piece.color == chess.BLACK:  # Black pieces offset by 6
                    layer += 6
                obs[layer, row, col] = 1.0 if piece.color == chess.WHITE else -1.0

        return obs

    def set_state(self, new_board):
        """
        Updates the internal board state of the environment.

        Args:
            new_board (list[list[str]]): The new board state represented as an 8x8 list of piece strings.
                                         Each string represents a piece, e.g., 'wP' for white pawn, '--' for empty square.
        """
        import chess  # Ensure the `chess` library is imported

        self.board = chess.Board()  # Reset the board
        self.board.clear_board()  # Clear the board to start from a clean slate

        # Recreate the board state
        for row in range(8):
            for col in range(8):
                piece = new_board[row][col]
                if piece != '--':  # If there's a piece, add it to the board
                    piece_type = piece[1].lower()  # Convert to lowercase for compatibility with `python-chess`
                    color = chess.WHITE if piece[0] == 'w' else chess.BLACK
                    self.board.set_piece_at(row * 8 + col,
                                            chess.Piece.from_symbol(piece_type.upper() if color else piece_type))

    def get_reward(self):
        """Calculates the reward based on the current board state."""
        # Use evaluate_board to calculate the reward
        reward = self.evaluate_board(self.board, self.previous_board)
        return reward

    def render(self, mode='human'):
        """Renders the current board state."""
        if mode == 'human':
            print(self.board)  # Print the board to console
        else:
            # You could extend this to support other render modes (like 'rgb_array')
            pass

    # Function to calculate rewards based on custom evaluation metrics
    def evaluate_board(self, current_board, previous_board=None):
        """
        Combines multiple evaluation functions to calculate the overall reward.
        """
        reward = 0
        reward += self.evaluate_material()
        reward += self.evaluate_piece_activity()
        reward += self.evaluate_king_safety()
        reward += self.evaluate_positional_advantage()
        reward += self.evaluate_piece_coordination()
        reward += self.evaluate_tempo()

        if previous_board:
            reward += self.evaluate_move_quality(previous_board, current_board)

        reward += self.evaluate_game_result(current_board)

        return reward / 10  # Normalize rewards to avoid large fluctuations

    def evaluate_material(self):
        """
        Evaluates material advantage.
        Includes piece-square tables to enhance context.
        """
        piece_values = {'P': 1, 'N': 3.2, 'B': 3.3, 'R': 5, 'Q': 9}
        piece_square_table = {
            'P': [  # Pawns
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1],
                [0.05, 0.05, 0.1, 0.275, 0.275, 0.1, 0.05, 0.05],
                [0, 0, 0, 0.25, 0.25, 0, 0, 0],
                [0.05, -0.05, -0.1, 0, 0, -0.1, -0.05, 0.05],
                [0.05, 0.1, 0.1, -0.2, -0.2, 0.1, 0.1, 0.05],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            # Similar tables can be added for other pieces: 'N', 'B', 'R', 'Q', and 'K'
        }

        reward = 0
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                value = piece_values.get(piece.symbol().upper(), 0)
                row, col = divmod(i, 8)
                if piece.color == chess.WHITE:
                    reward += value + piece_square_table.get(piece.symbol().upper(), [[0] * 8] * 8)[row][col]
                else:
                    reward -= value + piece_square_table.get(piece.symbol().upper(), [[0] * 8] * 8)[7 - row][col]
        return reward

    def evaluate_piece_activity(self):
        """
        Evaluates piece activity by rewarding mobility and central control.
        """
        activity_reward = 0
        center_squares = {chess.D4, chess.E4, chess.D5, chess.E5}
        for square, piece in self.board.piece_map().items():
            legal_moves = list(self.board.legal_moves)
            activity_reward += 0.1 * len(legal_moves)  # Reward mobility
            if square in center_squares:
                activity_reward += 0.5  # Reward central control
        return activity_reward

    def evaluate_king_safety(self):
        """
        Evaluates king safety based on castling and exposure.
        """
        reward = 0
        for color in [chess.WHITE, chess.BLACK]:
            king_square = self.board.king(color)
            if king_square:
                if king_square in [chess.G1, chess.C1, chess.G8, chess.C8]:  # Castled positions
                    reward += 1 if color == chess.WHITE else -1
                else:
                    adjacent_squares = list(self.board.attacks(king_square))
                    for square in adjacent_squares:
                        if not self.board.is_attacked_by(not color, square):
                            reward += 0.1 if color == chess.WHITE else -0.1
        return reward

    def evaluate_positional_advantage(self):
        """
        Evaluates positional advantages like outposts and open files.
        """
        reward = 0
        open_files = [i for i in range(8) if all(self.board.piece_at(chess.square(i, j)) is None for j in range(8))]
        for square, piece in self.board.piece_map().items():
            if piece.symbol().upper() == 'R' and chess.square_file(square) in open_files:
                reward += 0.5 if piece.color == chess.WHITE else -0.5  # Rook on open file
        return reward

    def evaluate_piece_coordination(self):
        """
        Evaluates piece coordination (support between pieces).
        """
        reward = 0
        for square, piece in self.board.piece_map().items():
            for attack in self.board.attacks(square):
                if self.board.piece_at(attack) and self.board.piece_at(attack).color == piece.color:
                    reward += 0.1 if piece.color == chess.WHITE else -0.1
        return reward

    def evaluate_tempo(self):
        """
        Rewards quick development of pieces.
        """
        reward = 0
        developed_pieces = 0
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece and piece.color == chess.WHITE and i in range(16, 48):  # Developed zone
                developed_pieces += 1
        reward += 0.2 * developed_pieces  # Reward based on number of developed pieces
        return reward

    def evaluate_move_quality(self, previous_board, current_board):
        """
        Compares AI move to Stockfish's best move (optional).
        """
        stockfish.set_fen_position(previous_board.fen())
        best_move = stockfish.get_best_move()
        ai_move = current_board.last_move
        if ai_move == best_move:
            return 1  # Perfect move
        else:
            return -1  # Sub-optimal move

    def evaluate_game_result(self, board):
        """
        Evaluates the game result.
        """
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                return -10  # Black wins
            else:
                return 10  # White wins
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_fifty_moves():
            return 0  # Draw
        return 0  # No terminal result

