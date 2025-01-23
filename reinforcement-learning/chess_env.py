import chess
import numpy as np
import settings
from gymnasium import Env
from gymnasium import spaces
from stockfish import Stockfish  # Assuming Stockfish is installed and available for move quality

# Initialize Stockfish (optional for evaluating move quality)
stockfish = Stockfish(path=settings.stockfish_path)


class ChessEnv(Env):
    def __init__(self, render_mode=None):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.previous_board = None

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
        self.previous_board = None
        return self.get_observation(), {}

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        move = legal_moves[action % len(legal_moves)]
        self.previous_board = self.board.copy()
        self.board.push(move)

        done = self.board.is_game_over()
        reward = self.get_reward()
        info = {"episode": {"r": reward, "l": len(self.board.move_stack)}} if done else {}

        return self.get_observation(), reward, done, False, info

    def get_observation(self):
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
                if piece.color == chess.BLACK:
                    layer += 6
                obs[layer, row, col] = 1.0

        return obs

    def render(self, mode='human'):
        """Renders the current board state."""
        if mode == 'human':
            print(self.board)
        else:
            pass  # Extend for other render modes if needed

    def set_state(self, new_board):
        """
        Updates the internal board state of the environment.

        Args:
            new_board (list[list[str]]): The new board state represented as an 8x8 list of piece strings.
                                         Each string represents a piece, e.g., 'wP' for white pawn, '--' for empty square.
        """
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
        reward = self.evaluate_board(self.board, self.previous_board)

        # Evaluate trade if the last move involved a capture
        if self.board.move_stack:
            last_move = self.board.move_stack[-1]
            reward += self.evaluate_trade(last_move)

        if not np.isfinite(reward):
            print(f"Invalid reward detected: {reward}. Resetting to 0.")
            reward = 0

        return reward


    def evaluate_trade(self, move):
        """
        Evaluates whether a trade is favorable based on piece values and protection.

        Args:
            move (chess.Move): The move to evaluate.

        Returns:
            float: Positive value for favorable trades, negative for unfavorable trades, 0 for neutral.
        """
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

        # Get the pieces involved in the trade
        moving_piece = self.board.piece_at(move.from_square)
        captured_piece = self.board.piece_at(move.to_square)

        if not moving_piece:
            return 0  # No piece is being moved (shouldn't happen with valid moves)

        # If there's no captured piece, it's not a trade
        if not captured_piece:
            return 0

        # Evaluate the trade
        moving_piece_value = piece_values.get(moving_piece.piece_type, 0)
        captured_piece_value = piece_values.get(captured_piece.piece_type, 0)

        # Check if the captured piece is protected
        is_captured_piece_protected = any(
            self.board.piece_at(attacker) and
            self.board.piece_at(attacker).color != captured_piece.color
            for attacker in self.board.attackers(not captured_piece.color, move.to_square)
        )

        # Reward or penalize based on the trade
        trade_value = captured_piece_value - moving_piece_value
        if is_captured_piece_protected:
            trade_value -= moving_piece_value  # Account for losing the moving piece after the trade

        return trade_value


    def evaluate_board(self, current_board, previous_board=None):
        """
        Combines multiple evaluation functions to calculate the overall reward.
        """
        reward = 0
        reward += self.evaluate_material()
        reward += self.evaluate_king_safety()
        reward += self.evaluate_positional_advantage()
        reward += self.evaluate_piece_coordination()
        reward += self.evaluate_tempo()
        reward += self.evaluate_center_control()  # Add center control evaluation

        # if previous_board:
        #     reward += self.evaluate_move_quality(previous_board, current_board)

        reward += self.evaluate_game_result(current_board)

        return reward / 10  # Normalize rewards to avoid large fluctuations

    def evaluate_material(self):
        """
        Evaluates material advantage with added mobility and piece-square tables for context.
        """
        piece_values = {'P': 1, 'N': 3.2, 'B': 3.3, 'R': 5, 'Q': 9}
        piece_square_tables = {
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
            'N': [  # Knights
                [-0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5],
                [-0.4, -0.2, 0, 0, 0, 0, -0.2, -0.4],
                [-0.3, 0, 0.1, 0.15, 0.15, 0.1, 0, -0.3],
                [-0.3, 0.05, 0.15, 0.2, 0.2, 0.15, 0.05, -0.3],
                [-0.3, 0, 0.15, 0.2, 0.2, 0.15, 0, -0.3],
                [-0.3, 0.05, 0.1, 0.15, 0.15, 0.1, 0.05, -0.3],
                [-0.4, -0.2, 0, 0.05, 0.05, 0, -0.2, -0.4],
                [-0.5, -0.4, -0.3, -0.3, -0.3, -0.3, -0.4, -0.5],
            ],
            'B': [  # Bishops
                [-0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2],
                [-0.1, 0, 0, 0, 0, 0, 0, -0.1],
                [-0.1, 0, 0.05, 0.1, 0.1, 0.05, 0, -0.1],
                [-0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, -0.1],
                [-0.1, 0, 0.1, 0.1, 0.1, 0.1, 0, -0.1],
                [-0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1],
                [-0.1, 0.05, 0, 0, 0, 0, 0.05, -0.1],
                [-0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2],
            ],
            'R': [  # Rooks
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],
                [-0.05, 0, 0, 0, 0, 0, 0, -0.05],
                [-0.05, 0, 0, 0, 0, 0, 0, -0.05],
                [-0.05, 0, 0, 0, 0, 0, 0, -0.05],
                [-0.05, 0, 0, 0, 0, 0, 0, -0.05],
                [-0.05, 0, 0, 0, 0, 0, 0, -0.05],
                [0, 0, 0, 0.05, 0.05, 0, 0, 0],
            ],
            'Q': [  # Queens
                [-0.2, -0.1, -0.1, -0.05, -0.05, -0.1, -0.1, -0.2],
                [-0.1, 0, 0, 0, 0, 0, 0, -0.1],
                [-0.1, 0, 0.05, 0.05, 0.05, 0.05, 0, -0.1],
                [-0.05, 0, 0.05, 0.05, 0.05, 0.05, 0, -0.05],
                [0, 0, 0.05, 0.05, 0.05, 0.05, 0, -0.05],
                [-0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0, -0.1],
                [-0.1, 0, 0.05, 0, 0, 0, 0, -0.1],
                [-0.2, -0.1, -0.1, -0.05, -0.05, -0.1, -0.1, -0.2],
            ],
            'K': [  # Kings (middle-game)
                [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
                [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
                [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
                [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
                [-0.2, -0.3, -0.3, -0.4, -0.4, -0.3, -0.3, -0.2],
                [-0.1, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1],
                [0.2, 0.2, 0, 0, 0, 0, 0.2, 0.2],
                [0.2, 0.3, 0.1, 0, 0, 0.1, 0.3, 0.2],
            ],
        }

        reward = 0
        for square, piece in self.board.piece_map().items():
            value = piece_values.get(piece.symbol().upper(), 0)
            row, col = divmod(square, 8)
            mobility = len(list(self.board.legal_moves))
            piece_table = piece_square_tables.get(piece.symbol().upper(), [[0] * 8] * 8)

            if piece.color == chess.WHITE:
                reward += value + piece_table[row][col] + 0.01 * mobility
            else:
                reward -= value + piece_table[7 - row][col] + 0.01 * mobility

        return reward

    def evaluate_center_control(self):
        """
        Rewards controlling the center of the board.
        Central squares are d4, e4, d5, and e5, with slightly less emphasis on the larger center.
        """
        central_squares = {chess.D4, chess.E4, chess.D5, chess.E5}  # Most critical center
        extended_center = {chess.C3, chess.C4, chess.C5, chess.C6,
                           chess.D3, chess.D6, chess.E3, chess.E6,
                           chess.F3, chess.F4, chess.F5, chess.F6}  # Larger center

        reward = 0
        for square in central_squares:
            if self.board.is_attacked_by(chess.WHITE, square):
                reward += 0.5  # Reward for controlling key central squares
            if self.board.is_attacked_by(chess.BLACK, square):
                reward -= 0.5

        for square in extended_center:
            if self.board.is_attacked_by(chess.WHITE, square):
                reward += 0.25  # Lesser reward for extended center
            if self.board.is_attacked_by(chess.BLACK, square):
                reward -= 0.25

        return reward

    def evaluate_king_safety(self):
        """
        Evaluates king safety based on castling, exposure, and pawn protection.
        """
        reward = 0

        # Define castling positions
        castled_positions = {
            chess.WHITE: [chess.G1, chess.C1],  # White kingside and queenside
            chess.BLACK: [chess.G8, chess.C8]  # Black kingside and queenside
        }

        # Check castling status
        for color in [chess.WHITE, chess.BLACK]:
            king_square = self.board.king(color)
            if king_square in castled_positions[color]:
                reward += 1 if color == chess.WHITE else -1  # Reward castling

        # Penalize exposed kings
        for color in [chess.WHITE, chess.BLACK]:
            king_square = self.board.king(color)
            if king_square:
                adjacent_squares = list(self.board.attacks(king_square))
                for square in adjacent_squares:
                    if self.board.is_attacked_by(not color, square):  # If an enemy piece attacks an adjacent square
                        reward -= 0.1 if color == chess.WHITE else -0.1

        # Encourage pawn cover around the king
        pawn_cover_positions = {
            chess.WHITE: [chess.G2, chess.F2, chess.H2] if self.board.king(chess.WHITE) == chess.G1 else [chess.C2,
                                                                                                          chess.B2,
                                                                                                          chess.D2],
            chess.BLACK: [chess.G7, chess.F7, chess.H7] if self.board.king(chess.BLACK) == chess.G8 else [chess.C7,
                                                                                                          chess.B7,
                                                                                                          chess.D7]
        }

        for color in [chess.WHITE, chess.BLACK]:
            for square in pawn_cover_positions[color]:
                if self.board.piece_at(square) and self.board.piece_at(square).symbol().upper() == 'P':
                    reward += 0.2 if color == chess.WHITE else -0.2  # Reward for pawns protecting the king

        return reward

    def evaluate_positional_advantage(self):
        """
        Evaluates positional advantages, including open files, semi-open files, outposts,
        and penalties for positional weaknesses like doubled pawns.
        """
        reward = 0

        # Define open and semi-open files
        open_files = [i for i in range(8) if all(self.board.piece_at(chess.square(i, j)) is None for j in range(8))]
        semi_open_files = [
            i for i in range(8)
            if any(self.board.piece_at(chess.square(i, j)) is not None for j in range(8)) and
               not any(self.board.piece_at(chess.square(i, j)) is not None and
                       self.board.piece_at(chess.square(i, j)).color == chess.WHITE for j in range(8))
        ]

        for square, piece in self.board.piece_map().items():
            piece_type = piece.symbol().upper()
            color = piece.color

            # Rook on open or semi-open files
            if piece_type == 'R':
                file = chess.square_file(square)
                if file in open_files:
                    reward += 0.5 if color == chess.WHITE else -0.5  # Rook on open file
                elif file in semi_open_files:
                    reward += 0.25 if color == chess.WHITE else -0.25  # Rook on semi-open file

            # Knights on outposts (protected and cannot be attacked by pawns)
            if piece_type == 'N':
                if self.is_outpost(square, color):
                    reward += 0.3 if color == chess.WHITE else -0.3

            # Penalty for doubled pawns
            if piece_type == 'P':
                file = chess.square_file(square)
                pawns_in_file = [
                    chess.square(file, rank)
                    for rank in range(8)
                    if self.board.piece_at(chess.square(file, rank)) and
                       self.board.piece_at(chess.square(file, rank)).symbol().upper() == 'P' and
                       self.board.piece_at(chess.square(file, rank)).color == color
                ]
                if len(pawns_in_file) > 1:  # Doubled pawns
                    reward -= 0.2 if color == chess.WHITE else -0.2

        return reward

    def is_outpost(self, square, color):
        """
        Checks if a knight's position is an outpost.
        An outpost is a square protected by a pawn and cannot be attacked by enemy pawns.
        """
        pawn_color = chess.WHITE if color == chess.WHITE else chess.BLACK
        opponent_color = not pawn_color

        # Define pawn movement directions
        forward_left = square + (-9 if pawn_color == chess.WHITE else 7)
        forward_right = square + (-7 if pawn_color == chess.WHITE else 9)

        # Check if the square is protected by a pawn
        pawns_protecting = [forward_left, forward_right]
        is_protected = any(
            0 <= sq < 64 and
            self.board.piece_at(sq) and self.board.piece_at(sq).symbol().upper() == 'P' and
            self.board.piece_at(sq).color == color
            for sq in pawns_protecting
        )

        # Check if the square cannot be attacked by enemy pawns
        backward_left = square + (7 if pawn_color == chess.WHITE else -9)
        backward_right = square + (9 if pawn_color == chess.WHITE else -7)
        opponent_pawns_threatening = [backward_left, backward_right]
        is_safe = all(
            not (0 <= sq < 64 and
                 self.board.piece_at(sq) and
                 self.board.piece_at(sq).symbol().upper() == 'P' and
                 self.board.piece_at(sq).color == opponent_color)
            for sq in opponent_pawns_threatening
        )

        return is_protected and is_safe


    def evaluate_piece_coordination(self):
        """
        Evaluates piece coordination by rewarding mutual support, coordinated attacks, and synergy.
        """
        reward = 0

        # Reward mutual support between pieces
        for square, piece in self.board.piece_map().items():
            for attack in self.board.attacks(square):
                defender = self.board.piece_at(attack)
                if defender and defender.color == piece.color:
                    reward += 0.1 if piece.color == chess.WHITE else -0.1  # Reward mutual defense

        # Reward coordinated attacks on the same square
        attacked_squares = {}
        for square, piece in self.board.piece_map().items():
            if piece.color == chess.WHITE:
                for attack in self.board.attacks(square):
                    attacked_squares[attack] = attacked_squares.get(attack, 0) + 1
            else:
                for attack in self.board.attacks(square):
                    attacked_squares[attack] = attacked_squares.get(attack, 0) - 1

        for square, count in attacked_squares.items():
            if count > 1:  # White pieces attacking the same square
                reward += 0.05 * (count - 1)  # Incremental reward for multiple attacks
            elif count < -1:  # Black pieces attacking the same square
                reward -= 0.05 * (-count - 1)

        # Reward synergy, such as rooks on the same file
        for color in [chess.WHITE, chess.BLACK]:
            rooks = [square for square, piece in self.board.piece_map().items()
                     if piece.symbol().upper() == 'R' and piece.color == color]
            for i, rook1 in enumerate(rooks):
                for rook2 in rooks[i + 1:]:
                    if chess.square_file(rook1) == chess.square_file(rook2):  # Rooks on the same file
                        reward += 0.2 if color == chess.WHITE else -0.2

        return reward

    def evaluate_tempo(self):
        """
        Rewards quick development of pieces and penalizes wasting tempo.
        """
        reward = 0

        # Minor pieces that have been developed (not on starting squares)
        minor_pieces = ['N', 'B']  # Knights and Bishops
        starting_squares = {
            chess.WHITE: [chess.B1, chess.G1, chess.C1, chess.F1],
            chess.BLACK: [chess.B8, chess.G8, chess.C8, chess.F8]
        }

        # Count developed minor pieces
        for color in [chess.WHITE, chess.BLACK]:
            for square in starting_squares[color]:
                piece = self.board.piece_at(square)
                if piece and piece.symbol().upper() in minor_pieces and piece.color == color:
                    reward -= 0.2 if color == chess.WHITE else -0.2  # Penalize undeveloped minor pieces

        # Reward moving pieces toward the center
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square, piece in self.board.piece_map().items():
            if piece.color == chess.WHITE:
                reward += 0.1 if square in center_squares else 0
            else:
                reward -= 0.1 if square in center_squares else 0

        # Penalize moving the same piece multiple times without justification
        move_log = self.board.move_stack[-10:]  # Look at the last 10 moves
        piece_move_count = {}
        for move in move_log:
            piece_move_count[move.from_square] = piece_move_count.get(move.from_square, 0) + 1

        for count in piece_move_count.values():
            if count > 1:  # Penalize repeated moves
                reward -= 0.05 * count

        return reward

    def evaluate_move_quality(self, previous_board, current_board):
        """
        Compares the AI move to Stockfish's best move and evaluates the move's quality
        based on evaluation scores.
        """
        # Set Stockfish to the previous board state
        stockfish.set_depth(5)
        stockfish.update_engine_parameters({"Threads": 1, "Hash": 128})
        stockfish.set_fen_position(previous_board.fen())

        # Get Stockfish's evaluation score before the AI move
        previous_eval = stockfish.get_evaluation()["value"]  # Can return centipawn scores or "mate in X"

        # Get Stockfish's best move
        best_move = stockfish.get_best_move()

        # Set Stockfish to the current board state after AI's move
        stockfish.set_fen_position(current_board.fen())

        # Get Stockfish's evaluation score after the AI's move
        ai_eval = stockfish.get_evaluation()["value"]

        # Reset Stockfish to the previous board and simulate the best move
        stockfish.set_fen_position(previous_board.fen())
        stockfish.make_moves_from_current_position([best_move])

        # Get Stockfish's evaluation score after the best move
        best_eval = stockfish.get_evaluation()["value"]

        # Evaluate the move quality
        if ai_eval == best_eval:
            return 1  # Perfect move
        elif ai_eval > previous_eval:
            return 0.5  # Good move, improved the position
        elif ai_eval < previous_eval:
            return -0.5  # Bad move, worsened the position
        else:
            return -1  # Severe blunder

    def evaluate_game_result(self, board):
        """
        Evaluates the game result, rewarding faster wins with higher scores.
        """
        move_count = len(self.board.move_stack)  # Count the number of moves played so far

        # Define base rewards
        win_base_reward = 7
        loss_base_reward = -10
        max_moves = 40  # Assumed maximum number of moves for a normal game
        early_win_bonus = (max_moves - move_count)/10  # The fewer the moves, the higher the bonus

        if self.board.is_checkmate():
            if self.board.turn == chess.WHITE:
                return loss_base_reward  # Black wins
            else:
                return win_base_reward + early_win_bonus  # White wins with bonus for early win
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_fifty_moves():
            return 0  # Draw
        return 0  # No terminal result


