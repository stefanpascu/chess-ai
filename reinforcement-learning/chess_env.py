import chess
import numpy as np
from gymnasium import Env, spaces
from numpy.f2py.auxfuncs import throw_error


class ChessEnv(Env):
    def __init__(self, render_mode=None, reward_scaling_factor=1.0):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.reward_scaling_factor = reward_scaling_factor

        self.all_possible_moves = self._generate_all_possible_moves()
        n_actions = len(self.all_possible_moves)

        self.action_space = spaces.Discrete(n_actions)

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1.0, shape=(18, 8, 8), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(n_actions,), dtype=np.bool_)
        })

        self.render_mode = render_mode
        self.result = None
        self.winner = None

    def _generate_all_possible_moves(self):
        moves = []
        for from_sq in chess.SQUARES:
            for to_sq in chess.SQUARES:
                move = chess.Move(from_sq, to_sq)
                moves.append(move.uci())
                if chess.square_rank(from_sq) == 6 and chess.square_rank(to_sq) == 7:
                    for prom in ['q', 'r', 'b', 'n']:
                        promo_move = chess.Move(from_sq, to_sq, promotion=chess.Piece.from_symbol(prom.upper()).piece_type)
                        moves.append(promo_move.uci())
                if chess.square_rank(from_sq) == 1 and chess.square_rank(to_sq) == 0:
                    for prom in ['q', 'r', 'b', 'n']:
                        promo_move = chess.Move(from_sq, to_sq, promotion=chess.Piece.from_symbol(prom.lower()).piece_type)
                        moves.append(promo_move.uci())
        # Remove duplicates while preserving order.
        moves = list(dict.fromkeys(moves))
        return moves

    def get_observation(self):
        planes = np.zeros((18, 8, 8), dtype=np.float32)
        piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                     chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                layer = piece_map[piece.piece_type]
                if piece.color == chess.BLACK:
                    layer += 6
                planes[layer, row, col] = 1.0

        castling_map = {'K': 12, 'Q': 13, 'k': 14, 'q': 15}
        if self.board.has_kingside_castling_rights(chess.WHITE):
            planes[castling_map['K'], :, :] = 1.0
        if self.board.has_queenside_castling_rights(chess.WHITE):
            planes[castling_map['Q'], :, :] = 1.0
        if self.board.has_kingside_castling_rights(chess.BLACK):
            planes[castling_map['k'], :, :] = 1.0
        if self.board.has_queenside_castling_rights(chess.BLACK):
            planes[castling_map['q'], :, :] = 1.0

        planes[16, :, :] = self.board.halfmove_clock / 100.0
        if self.board.ep_square is not None:
            row, col = divmod(self.board.ep_square, 8)
            planes[17, row, col] = 1.0

        return planes

    def get_action_mask(self):
        """Returns a binary mask indicating legal moves for the current board position."""

        # 🚨 If game is over, return all zeros
        if self.board.is_game_over():
            print(f"🚨 Game is over at FEN: {self.board.fen()} | Returning all-zero action mask.")
            return np.zeros(len(self.all_possible_moves), dtype=np.bool_)

        # Initialize mask with zeros
        action_mask = np.zeros(len(self.all_possible_moves), dtype=np.bool_)
        legal_moves = self.get_legal_moves()

        for i, move in enumerate(self.all_possible_moves):
            if move in legal_moves:
                action_mask[i] = 1  # Mark legal moves as 1

        # print("(Inside get_action_mask)Action mask contains non-binary values: ", action_mask)

        return action_mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        self.result = None
        self.winner = None
        obs = {
            "observation": self.get_observation(),
            "action_mask": self.get_action_mask()
        }
        return obs, {}

    def step(self, action):
        move_uci = self.all_possible_moves[action]
        move = chess.Move.from_uci(move_uci)

        if move not in self.board.legal_moves:
            print(f"🚨 ERROR: Move {move_uci} is not legal in this state!")
            print(f"Legal moves: {[m.uci() for m in self.board.legal_moves]}")
            raise ValueError(
                f"Chosen move {move_uci} is illegal in this state. "
                f"Legal moves count: {len(list(self.board.legal_moves))}"
            )

        self.board.push(chess.Move.from_uci(move_uci))

        done = self.board.is_game_over()
        reward = (self._evaluate_result() * self.reward_scaling_factor
                  if done else self._evaluate_material_balance())

        obs = {
            "observation": self.get_observation(),
            "action_mask": self.get_action_mask() if not done else np.zeros(len(self.all_possible_moves),
                                                                            dtype=np.bool_)
        }
        # print(f"(Inside step) Correct action mask:\n{obs['action_mask']}")

        if done and np.sum(obs["action_mask"]) != 0:
            print("🚨 ERROR: Game over but action mask is not zeros!")

        return obs, reward, done, False, {}

    def render(self, mode="human"):
        if mode == "human":
            print(self.board)

    def set_state(self, new_board):
        self.board = chess.Board(fen=new_board)

    def adjudicate(self):
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

    def _evaluate_result(self):
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        elif (self.board.is_stalemate() or
              self.board.is_insufficient_material() or
              self.board.can_claim_fifty_moves()):
            return 0
        else:
            return 0

    def _evaluate_material_balance(self):
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
        return [move.uci() for move in self.board.legal_moves]
