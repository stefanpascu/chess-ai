import chess
import negamax as ai

# Player settings. Set to True if a human is playing that color.
player_white = True
player_black = False

def main():
    board = chess.Board()

    while True:
        # Print the board to the console
        print(board, "\n")

        # Check for end of game
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            print(f"Checkmate! {winner} wins.")
            break
        if board.is_stalemate():
            print("Stalemate!")
            break
        if board.is_insufficient_material():
            print("Draw by insufficient material.")
            break

        human_turn = (board.turn == chess.WHITE and player_white) or \
                     (board.turn == chess.BLACK and player_black)

        if human_turn:
            # Ask the user for a move in UCI format, e.g. "e2e4"
            move_uci = input("Your move (in UCI, e.g. e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                print("Invalid UCI format. Try again.")
                continue

            if move not in board.legal_moves:
                print("Illegal move. Try again.")
                continue

            board.push(move)
        else:
            # Let your negamax AI choose a move
            legal_moves = list(board.legal_moves)
            ai_move = ai.find_best_move(board, legal_moves)
            if ai_move is None:
                ai_move = ai.find_random_move(legal_moves)

            print(f"AI plays: {ai_move}")
            board.push(ai_move)

if __name__ == "__main__":
    main()
