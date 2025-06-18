import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pygame as p
import numpy as np
import chess                                      # for Negamax engine
import chess_engine
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from chess_env import ChessEnv
import minmax_negamax.negamax as ai_negamax
import reinforcement_learning.rl_settings as settings

# ——— Load your RL model ———
model_path = settings.best_model_path if os.path.exists(settings.best_model_path) \
    else settings.reinforced_model_path
model = PPO.load(model_path)
env   = DummyVecEnv([lambda: ChessEnv()])

# ——— Pygame setup ———
p.init()
board_width = board_height = 680
dimension   = 8
sq_size     = board_height // dimension
max_fps     = 15
move_log_w  = 210
move_log_h  = board_height
colours = [p.Color('#EBEBD0'), p.Color('#769455')]
images  = {}

def load_images():
    pieces = ['bR','bN','bB','bQ','bK','bB','bN','bR','bP',
              'wR','wN','wB','wQ','wK','wB','wN','wR','wP']
    img_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'images'))
    for piece in pieces:
        path = os.path.join(img_dir, f'{piece}.png')
        images[piece] = p.transform.smoothscale(
            p.image.load(path).convert_alpha(),
            (sq_size, sq_size)
        )

def draw_board(screen):
    for r in range(dimension):
        for c in range(dimension):
            p.draw.rect(
                screen,
                colours[(r+c) % 2],
                p.Rect(c*sq_size, r*sq_size, sq_size, sq_size)
            )

def draw_pieces(screen, board):
    for r in range(dimension):
        for c in range(dimension):
            piece = board[r][c]
            if piece != '--':
                screen.blit(
                    images[piece],
                    p.Rect(c*sq_size, r*sq_size, sq_size, sq_size)
                )

def draw_move_log(screen, gs, font):
    area = p.Rect(board_width, 0, move_log_w, move_log_h)
    p.draw.rect(screen, p.Color('#2d2d2e'), area)
    y = 5
    moves = gs.move_log
    rows = []
    for i in range(0, len(moves), 2):
        s = f"{i//2+1}. {moves[i]} "
        if i+1 < len(moves): s += str(moves[i+1]) + " "
        rows.append(s)
    for i in range(0, len(rows), 2):
        line = rows[i] + (rows[i+1] if i+1 < len(rows) else "")
        txt = font.render(line, True, p.Color('whitesmoke'))
        screen.blit(txt, area.move(5, y))
        y += txt.get_height() + 2

def draw_game_state(screen, gs, font):
    draw_board(screen)
    draw_pieces(screen, gs.board)
    draw_move_log(screen, gs, font)

def animate_move(move, screen, board, clock):
    dr = move.end_row - move.start_row
    dc = move.end_column - move.start_column
    frames = (abs(dr) + abs(dc)) * 5
    for f in range(frames + 1):
        r = move.start_row + dr * f/frames
        c = move.start_column + dc * f/frames
        draw_board(screen)
        draw_pieces(screen, board)
        dest_col = colours[(move.end_row + move.end_column) % 2]
        dest_sq  = p.Rect(move.end_column*sq_size,
                          move.end_row*sq_size,
                          sq_size, sq_size)
        p.draw.rect(screen, dest_col, dest_sq)
        if move.piece_captured != '--':
            if move.is_en_passant_move:
                ep_r = move.end_row + (1 if move.piece_captured[0]=='b' else -1)
                dest_sq = p.Rect(move.end_column*sq_size,
                                 ep_r*sq_size,
                                 sq_size, sq_size)
            screen.blit(images[move.piece_captured], dest_sq)
        screen.blit(
            images[move.piece_moved],
            p.Rect(c*sq_size, r*sq_size, sq_size, sq_size)
        )
        p.display.flip()
        clock.tick(60)

def draw_endgame_text(screen, text):
    font = p.font.SysFont('Helvetica', 32, True, False)
    txt1 = font.render(text, True, p.Color('gray'), p.Color('mintcream'))
    x = board_width/2 - txt1.get_width()/2
    y = board_height/2 - txt1.get_height()/2
    screen.blit(txt1, (x, y))
    txt2 = font.render(text, True, p.Color('black'))
    screen.blit(txt2, (x+2, y+2))

if __name__ == '__main__':
    screen = p.display.set_mode((board_width + move_log_w, board_height))
    clock  = p.time.Clock()
    font   = p.font.SysFont('Arial', 14, False, False)
    load_images()

    # GameState for UI and logic
    game_state = chess_engine.GameState()
    # python-chess Board for Negamax
    pty_board  = chess.Board()

    move_made = False
    animate   = False
    game_over = False

    while True:
        for event in p.event.get():
            if event.type == p.QUIT:
                p.quit()
                exit()

        if not game_over:
            if game_state.white_to_move:
                # — White (Negamax) —
                legal_py = list(pty_board.legal_moves)
                best_py  = ai_negamax.find_best_move(pty_board, legal_py)

                # find matching GameState.Move by comparing squares
                sel = None
                for gm in game_state.get_valid_moves():
                    # convert gm to python-chess square ids
                    gm_from = chess.square(gm.start_column, 7 - gm.start_row)
                    gm_to   = chess.square(gm.end_column,   7 - gm.end_row)
                    if gm_from == best_py.from_square and gm_to == best_py.to_square:
                        sel = gm
                        break
                if sel is None:
                    raise RuntimeError(f"No matching move for {best_py}")

                game_state.make_move(sel)
                pty_board.push(best_py)

            else:
                # — Black (RL PPO) —
                obs = env.envs[0].get_observation()
                obs = np.flip(obs, axis=1)
                obs = obs[np.newaxis, ...]
                act_arr, _ = model.predict(obs, deterministic=True)
                idx        = int(act_arr[0])
                gm_moves   = game_state.get_valid_moves()
                sel        = gm_moves[idx % len(gm_moves)]

                game_state.make_move(sel)
                # also update python-chess board
                fs = chess.square(sel.start_column, 7 - sel.start_row)
                ts = chess.square(sel.end_column,   7 - sel.end_row)
                pty_board.push(chess.Move(fs, ts))

            move_made = animate = True

        if move_made:
            animate_move(game_state.move_log[-1], screen, game_state.board, clock)
            move_made = animate = False

        draw_game_state(screen, game_state, font)

        if game_state.checkmate or game_state.stalemate:
            game_over = True
            if game_state.stalemate:
                text = 'Stalemate'
            else:
                text = ('Black wins by checkmate'
                        if game_state.white_to_move
                        else 'White wins by checkmate')
            draw_endgame_text(screen, text)

        clock.tick(max_fps)
        p.display.flip()
