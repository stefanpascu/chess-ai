import random
import kr_settings as settings

def find_random_move(valid_moves):
    return random.choice(valid_moves)


def find_best_move(game_state, valid_moves):
    global next_move
    next_move = None
    random.shuffle(valid_moves)
    find_minimax_move_alphabeta(game_state, valid_moves, settings.set_depth, -settings.checkmate_points, settings.checkmate_points,
                                1 if game_state.white_to_move else -1)
    return next_move

def find_minimax_move_alphabeta(game_state, valid_moves, depth, alpha, beta, maximizing_player):
    global next_move
    if depth == 0:
        return score_board(game_state)

    if maximizing_player:
        max_score = -settings.checkmate_points
        for move in valid_moves:
            game_state.make_move(move)
            next_moves = game_state.get_valid_moves()
            score = find_minimax_move_alphabeta(game_state, next_moves, depth - 1, alpha, beta, False)
            if score > max_score:
                max_score = score
                if depth == settings.set_depth:
                    next_move = move
            game_state.undo_move()

            alpha = max(alpha, max_score)
            if beta <= alpha:
                break

        return max_score
    else:
        min_score = settings.checkmate_points
        for move in valid_moves:
            game_state.make_move(move)
            next_moves = game_state.get_valid_moves()
            score = find_minimax_move_alphabeta(game_state, next_moves, depth - 1, alpha, beta, True)
            if score < min_score:
                min_score = score
                if depth == settings.set_depth:
                    next_move = move
            game_state.undo_move()

            beta = min(beta, min_score)
            if beta <= alpha:
                break

        return min_score

def score_board(game_state):
    if game_state.checkmate:
        if game_state.white_to_move:
            return -settings.checkmate_points
        else:
            return settings.checkmate_points
    elif game_state.stalemate:
        return settings.stalemate_points

    score = 0
    for row in range(len(game_state.board)):
        for column in range(len(game_state.board)):
            if game_state.board[row][column][0] == 'w':
                score += settings.piece_scores[game_state.board[row][column][1]]
                score += settings.piece_positions[game_state.board[row][column]][row][column]
            elif game_state.board[row][column][0] == 'b':
                score -= settings.piece_scores[game_state.board[row][column][1]]
                score -= settings.piece_positions[game_state.board[row][column]][row][column]
    return score
