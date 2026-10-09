"""Terminal interface: main menu and Human vs Engine mode (REQ-003)."""

import chess

from chesscore import VERSION
from chesscore.engine import DEFAULT_DEPTH, MATE_SCORE, find_best_move

MENU = [
    ("play", "Play a game - Human vs Engine"),
    ("train", "Train the engine (self-play / external engine)"),
    ("stats", "View training statistics & rating history"),
    ("review", "Review a past game from PGN"),
    ("quit", "Exit ChessCore"),
]

FILES = "  a b c d e f g h"


def render_board(board, flipped=False):
    """Board as text lines, uppercase for White and lowercase for Black."""
    ranks = range(8) if flipped else range(7, -1, -1)
    files = range(7, -1, -1) if flipped else range(8)
    header = "  " + " ".join(chess.FILE_NAMES[f] for f in files)
    lines = [header]
    for rank in ranks:
        cells = []
        for file in files:
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece.symbol() if piece else ".")
        lines.append(f"{rank + 1} " + " ".join(cells))
    lines.append(header)
    return lines


def render_history(sans):
    """Move history in SAN, one line per full move."""
    lines = ["MOVE HISTORY", "-" * 24]
    for i in range(0, len(sans), 2):
        white = sans[i]
        black = sans[i + 1] if i + 1 < len(sans) else "..."
        lines.append(f"{i // 2 + 1:>3}. {white:<8}{black}")
    return lines


def format_score(score):
    if score is None:
        return "-"
    if abs(score) >= MATE_SCORE:
        return "mate" if score > 0 else "-mate"
    return f"{score / 100:+.1f}"


def render_screen(board, sans, flipped, last_score, depth):
    left = render_board(board, flipped)
    right = render_history(sans)
    height = max(len(left), len(right))
    left += [""] * (height - len(left))
    right += [""] * (height - len(right))
    width = len(FILES) + 4
    lines = [f"{l:<{width}}    {r}" for l, r in zip(left, right)]
    lines.append("")
    lines.append(f"eval {format_score(last_score)} depth {depth}")
    return "\n".join(lines)


def parse_move(board, text):
    """Accept coordinate notation (e2e4, e7e8q) or SAN (Nf3). None if illegal."""
    try:
        move = chess.Move.from_uci(text)
        if move in board.legal_moves:
            return move
        # Allow e7e8 without a promotion piece: promote to a queen.
        move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass
    try:
        return board.parse_san(text)
    except ValueError:
        return None


def ask(prompt, choices):
    while True:
        answer = input(prompt).strip().lower()
        if answer in choices:
            return answer
        print(f"Please answer one of: {', '.join(choices)}")


def ask_depth():
    while True:
        answer = input(f"search depth [{DEFAULT_DEPTH}] > ").strip()
        if not answer:
            return DEFAULT_DEPTH
        if answer.isdigit() and int(answer) >= 1:
            return int(answer)
        print("Depth must be a whole number of at least 1.")


def game_result_text(board):
    outcome = board.outcome()
    if outcome.winner is None:
        reason = outcome.termination.name.replace("_", " ").lower()
        return f"Draw ({reason}) {outcome.result()}"
    winner = "White" if outcome.winner == chess.WHITE else "Black"
    return f"{winner} wins by checkmate {outcome.result()}"


def play():
    color = ask("play as white or black? [w/b] > ", ["w", "b"])
    human = chess.WHITE if color == "w" else chess.BLACK
    depth = ask_depth()

    board = chess.Board()
    sans = []
    last_score = None

    while not board.is_game_over():
        print()
        print(render_screen(board, sans, human == chess.BLACK, last_score, depth))
        if board.turn == human:
            text = input("your move (e.g. e2e4, 'quit' to stop) > ").strip()
            if text.lower() == "quit":
                print("Game abandoned.")
                return
            move = parse_move(board, text)
            if move is None:
                print(f"Illegal or unreadable move: {text}")
                continue
        else:
            print("engine is thinking...")
            move, last_score = find_best_move(board, depth)
        sans.append(board.san(move))
        board.push(move)

    print()
    print(render_screen(board, sans, human == chess.BLACK, last_score, depth))
    print(game_result_text(board))


def print_menu():
    print()
    print("CHESSCORE")
    print(f"Terminal Chess Engine - v{VERSION} - 100% Python")
    print("-" * 60)
    for i, (name, description) in enumerate(MENU, start=1):
        print(f"[{i}] {name:<8} {description}")


def main():
    names = [name for name, _ in MENU]
    while True:
        print_menu()
        choice = input("> ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            choice = names[int(choice) - 1]
        if choice == "play":
            play()
        elif choice == "quit":
            return
        elif choice in names:
            print(f"'{choice}' is not implemented yet.")
        else:
            print(f"Unknown option: {choice}")
