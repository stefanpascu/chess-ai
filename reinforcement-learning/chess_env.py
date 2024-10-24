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
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8, 8), dtype=np.float32)

        # Store the render mode (e.g., 'human' for console printing, or others if needed)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state and returns an initial observation."""
        super().reset(seed=seed)
        self.board.reset()
        self.previous_board = None
        return self.get_observation(), {}

    def get_observation(self):
        """Converts the chess board state into an 8x8 numerical matrix."""
        obs = np.zeros((8, 8))
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                obs[i // 8][i % 8] = piece.piece_type
        return obs

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
        reward = 0
        reward += self.evaluate_material()
        reward += self.evaluate_piece_activity()
        reward += self.evaluate_king_safety()

        if previous_board:
            reward += self.evaluate_move_quality(previous_board, current_board)  # Evaluate the quality of the move
        reward += self.evaluate_game_result(current_board)  # Additional rewards for checkmate/draw results

        return reward

    # Material advantage (assign values to pieces)
    def evaluate_material(self):
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
        reward = 0
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                if piece.color == chess.WHITE:
                    reward += piece_values.get(piece.piece_type, 0)
                else:
                    reward -= piece_values.get(piece.piece_type, 0)
        return reward

    # Piece activity (reward for controlling center and moving pieces to active squares)
    def evaluate_piece_activity(self):
        activity_reward = 0
        center_squares = {chess.square(3, 3), chess.square(3, 4), chess.square(4, 3), chess.square(4, 4)}  # d4, e4, d5, e5
        for square, piece in self.board.piece_map().items():
            if square in center_squares:
                activity_reward += 0.5  # Reward for central control
        return activity_reward

    # King safety (reward for castling and keeping the king protected)
    def evaluate_king_safety(self):
        king_square = self.board.king(chess.WHITE)  # Get the position of the white king
        rook_squares = [chess.A1, chess.H1]  # Original positions of rooks for white
        castled = False

        # Check for white castling
        if king_square in (chess.C1, chess.G1):  # Kingside or queenside castling
            castled = True

        # Check for black king
        king_square_black = self.board.king(chess.BLACK)  # Get the position of the black king
        rook_squares_black = [chess.A8, chess.H8]  # Original positions of rooks for black

        # Check for black castling
        if king_square_black in (chess.C8, chess.G8):  # Kingside or queenside castling
            castled = True

        return 1 if castled else 0  # Reward for castling

    # Move quality (compare the AI move to the best Stockfish move, optional)
    def evaluate_move_quality(previous_board, current_board):
        stockfish.set_fen_position(previous_board.fen())
        best_move = stockfish.get_best_move()
        ai_move = current_board.last_move  # Assuming the ChessEnv class stores the last AI move
        if ai_move == best_move:
            return 1  # Perfect move
        else:
            return -1  # Sub-optimal move

    # Endgame result (reward for winning and penalize for losing)
    def evaluate_game_result(self, board):
        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                return -10  # Black wins
            else:
                return 10  # White wins
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_fifty_moves():
            return 0  # Draw
        return 0  # No terminal result
