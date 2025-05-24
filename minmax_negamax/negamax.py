import random
import chess
import minmax_negamax.kr_settings as settings

next_move = None

def find_random_move(valid_moves):
    return random.choice(valid_moves) if valid_moves else None

def find_best_move(board, valid_moves):
    global next_move
    next_move = None
    random.shuffle(valid_moves)
    find_negamax_move_alphabeta(
        board,
        valid_moves,
        settings.set_depth,
        -settings.checkmate_points,
        settings.checkmate_points,
        1 if board.turn == chess.WHITE else -1
    )
    return next_move

def find_negamax_move_alphabeta(board, valid_moves, depth, alpha, beta, turn_multiplier):
    global next_move
    if depth == 0 or board.is_game_over():
        return turn_multiplier * evaluate(board)

    max_score = -settings.checkmate_points
    for move in valid_moves:
        board.push(move)
        next_moves = list(board.legal_moves)
        score = -find_negamax_move_alphabeta(
            board,
            next_moves,
            depth - 1,
            -beta,
            -alpha,
            -turn_multiplier
        )
        board.pop()

        if score > max_score:
            max_score = score
            if depth == settings.set_depth:
                next_move = move

        alpha = max(alpha, max_score)
        if alpha >= beta:
            break

    return max_score

def evaluate(board):
    if board.is_checkmate():
        return -settings.checkmate_points if board.turn == chess.WHITE else settings.checkmate_points
    if board.is_stalemate():
        return settings.stalemate_points

    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            symbol = piece.symbol()
            color = 'w' if piece.color == chess.WHITE else 'b'
            p_type = symbol.upper()
            piece_score = settings.piece_scores[p_type]
            key = f"{color}{p_type}"
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            pos_score = settings.piece_positions[key][rank][file]
            if color == 'w':
                score += piece_score + pos_score
            else:
                score -= piece_score + pos_score
    return score
