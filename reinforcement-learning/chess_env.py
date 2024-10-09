import chess
import numpy as np
from gymnasium import Env
from gymnasium import spaces


class ChessEnv(Env):
    def __init__(self, render_mode=None):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()

        # Define the action space and observation space
        self.action_space = spaces.Discrete(4672)  # Adjust for all possible chess moves
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8, 8), dtype=np.float32)

        # Store the render mode (e.g., 'human' for console printing, or others if needed)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state and returns an initial observation."""
        super().reset(seed=seed)
        self.board.reset()
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
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.WHITE else -1
        if self.board.is_stalemate():
            return 0
        return 0

    def render(self, mode='human'):
        """Renders the current board state."""
        if mode == 'human':
            print(self.board)  # Print the board to console
        else:
            # You could extend this to support other render modes (like 'rgb_array')
            pass
