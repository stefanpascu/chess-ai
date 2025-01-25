import chess
import numpy as np
from gymnasium import Env, spaces

class ChessEnv(Env):
    def __init__(self, render_mode=None, reward_scaling_factor=1.0):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.reward_scaling_factor = reward_scaling_factor

        # Observation space for NatureCNN (12 channels, 8x8 board)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12, 8, 8), dtype=np.float64
        )

        # Action space for discrete moves
        self.action_space = spaces.Discrete(4672)

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        return self.get_observation(), {}

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
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
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_fifty_moves():
            return 0
        return 0  # Intermediate state

    def render(self, mode="human"):
        if mode == "human":
            print(self.board)
