import chess
import numpy as np
from gymnasium import Env, spaces

class ChessEnv(Env):
    def __init__(self, render_mode=None, reward_scaling_factor=1.0):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.reward_scaling_factor = reward_scaling_factor

        # Observation space: (18, 8, 8) representation
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(18, 8, 8), dtype=np.float64
        )

        self.all_possible_moves = [move.uci() for move in chess.Board().legal_moves]

        # Action space: 4672 discrete moves (all theoretical moves)
        # self.action_space = spaces.Discrete(4672)

        # Initialize action space (will be dynamically updated)
        self.action_space = None
        self.update_action_space()

        self.render_mode = render_mode
        self.result = None
        self.winner = None

    def update_action_space(self):
        """
        Dynamically update the action space based on the number of legal moves.
        """
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the starting position.
        """
        super().reset(seed=seed)
        self.board.reset()
        self.result = None
        self.winner = None
        self.update_action_space()  # Update action space
        return self.get_observation(), {}

    def step(self, action):
        """
        Execute a move based on the provided action index and update the environment state.
        :param action: Integer index of a legal move for the current board state.
        """
        # Get the list of legal moves for the current state
        legal_moves = list(self.board.legal_moves)

        if self.board.is_check():
            print("Legal moves: ", legal_moves)
            print("Number of legal moves: ", len(legal_moves))
            print("Action: ", action)

        if action >= len(legal_moves):
            print("Board: \n")
            print(self.board)
            raise ValueError(f"Invalid action {action}. Total legal moves: {len(legal_moves)}.")

        # Get the selected move and execute it
        move = legal_moves[action]
        self.board.push(move)

        # Check if the game is over
        done = self.board.is_game_over()
        reward = self._evaluate_result() * self.reward_scaling_factor if done else self._evaluate_material_balance()

        # Update action space for the next step
        self.update_action_space()

        return self.get_observation(), reward, done, False, {}

    def get_observation(self):
        """
        Convert the current board state into (18, 8, 8) neural network input planes.
        """
        planes = np.zeros((18, 8, 8), dtype=np.float32)

        # Piece planes (12 channels)
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                layer = piece_map[piece.piece_type]
                if piece.color == chess.BLACK:
                    layer += 6
                planes[layer, row, col] = 1.0

        # Castling rights (4 channels)
        castling_map = {'K': 12, 'Q': 13, 'k': 14, 'q': 15}
        castling_map = {'K': 12, 'Q': 13, 'k': 14, 'q': 15}
        if self.board.has_kingside_castling_rights(chess.WHITE):
            planes[castling_map['K']] = 1.0
        if self.board.has_queenside_castling_rights(chess.WHITE):
            planes[castling_map['Q']] = 1.0
        if self.board.has_kingside_castling_rights(chess.BLACK):
            planes[castling_map['k']] = 1.0
        if self.board.has_queenside_castling_rights(chess.BLACK):
            planes[castling_map['q']] = 1.0

        # Fifty-move rule (1 channel)
        planes[16] = self.board.halfmove_clock / 100.0

        # En passant (1 channel)
        if self.board.ep_square is not None:
            row, col = divmod(self.board.ep_square, 8)
            planes[17, row, col] = 1.0

        return planes

    def _evaluate_result(self):
        """
        Evaluate the game result to assign rewards.
        """
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_fifty_moves():
            return 0  # Draw
        else:
            return 0  # Intermediate state

    def render(self, mode="human"):
        """
        Render the current board state.
        """
        if mode == "human":
            print(self.board)

    def set_state(self, new_board):
        """
        Set the board state to a specific configuration.
        """
        self.board = chess.Board(fen=new_board)

    def adjudicate(self):
        """
        Adjudicate the result based on a heuristic evaluation.
        """
        score = self._evaluate_material_balance()
        if abs(score) < 0.01:
            self.result = "1/2-1/2"
            self.winner = None
        elif score > 0:
            self.result = "1-0"
            self.winner = chess.WHITE
        else:
            self.result = "0-1"
            self.winner = chess.BLACK

    def _evaluate_material_balance(self):
        """
        Heuristically evaluate the material balance of the board.
        """
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.25,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        balance = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                balance += value if piece.color == chess.WHITE else -value
        return balance

    def get_legal_moves(self):
        """
        Return a list of all legal moves in UCI notation.
        """
        return [move.uci() for move in self.board.legal_moves]
