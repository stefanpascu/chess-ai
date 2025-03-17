import chess
import numpy as np
from gymnasium import Env, spaces

class ChessEnv(Env):
    def __init__(self, render_mode=None, reward_scaling_factor=1.0):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.reward_scaling_factor = reward_scaling_factor

        # Observation space: 12 channels (for piece types), 8x8 board
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12, 8, 8), dtype=np.float64
        )

        # Action space: 4672 possible moves (upper bound of chess move space)
        self.action_space = spaces.Discrete(4672)

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        return self.get_observation(), {}

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 0:
            return self.get_observation(), 0, True, False, {}

        move = legal_moves[action % len(legal_moves)]
        self.board.push(move)

        done = self.board.is_game_over()
        reward = self.get_game_result() * self.reward_scaling_factor
        return self.get_observation(), reward, done, False, {}

    def get_observation(self):
        obs = np.zeros((12, 8, 8), dtype=np.float32)
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                row, col = divmod(i, 8)
                layer = piece_map[piece.piece_type]
                if piece.color == chess.BLACK:
                    layer += 6
                obs[layer, row, col] = 1.0

        return obs

    def get_game_result(self):
        """Provides a shaped reward based on the game result."""
        if self.board.is_checkmate():
            return 1000 if self.board.turn == chess.BLACK else -1000  # Checkmate
        elif self.board.is_stalemate():
            return 0  # Stalemate is neutral
        elif self.board.is_insufficient_material() or self.board.can_claim_fifty_moves():
            return 0  # Draws are neutral
        return self.evaluate_board() * self.reward_scaling_factor  # Intermediate state

    def evaluate_board(self):
        """Returns an evaluation score based on material, piece positioning, and other factors."""
        return (self.evaluate_material() +
                self.mobility_bonus() +
                self.threats_and_defenses() +
                self.king_safety())

    def evaluate_material(self):
        """Evaluates material advantage."""
        piece_scores = {
            chess.PAWN: 1.0, chess.KNIGHT: 3.2, chess.BISHOP: 3.3,
            chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 200.0
        }
        score = 0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_scores.get(piece.piece_type, 0)
                score += value if piece.color == chess.WHITE else -value

        return score

    def mobility_bonus(self):
        """Gives a small bonus for having more legal moves."""
        white_moves = len(list(self.board.legal_moves)) if self.board.turn else 0
        self.board.push(chess.Move.null())
        black_moves = len(list(self.board.legal_moves)) if not self.board.turn else 0
        self.board.pop()

        return (white_moves - black_moves) * 0.1  # Small impact on score

    def threats_and_defenses(self):
        """Encourages AI to attack high-value pieces and defend its own pieces."""
        score = 0
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 200}

        for move in self.board.legal_moves:
            if self.board.is_capture(move):  # If it's a capture move
                captured_piece = self.board.piece_at(move.to_square)
                if captured_piece:
                    piece_value = piece_values.get(captured_piece.piece_type, 0)
                    if self.board.turn == chess.WHITE:
                        score += piece_value
                    else:
                        score -= piece_value
        return score * 0.2  # Moderate impact

    def king_safety(self):
        """Rewards king safety by penalizing open files near the king."""
        king_square = self.board.king(self.board.turn)
        if king_square is None:
            return 0  # King not found (should never happen)
        row, col = divmod(king_square, 8)

        # Penalty if the king is too exposed
        penalty = -1.5 if row < 3 or row > 6 else 0
        return penalty if self.board.turn == chess.WHITE else -penalty

    def render(self, mode="human"):
        if mode == "human":
            print(self.board)

    def set_state(self, new_board):
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
