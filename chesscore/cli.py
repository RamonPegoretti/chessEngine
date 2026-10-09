"""Terminal interface: main menu and Human vs Engine mode (REQ-003)."""

import os
import re
import sys

import chess

from chesscore import VERSION
from chesscore.engine import DEFAULT_DEPTH, MATE_SCORE, find_best_move
from chesscore import engine
from chesscore.storage import Storage
from chesscore.train import MIN_POSITIONS, train as run_training
from chesscore.uci import ExternalEngine, match_result, play_match_game

MENU = [
    ("play", "Play a game - Human vs Engine"),
    ("match", "Watch ChessCore vs an external engine (Stockfish)"),
    ("train", "Train the engine (self-play / external engine)"),
    ("stats", "View training statistics & rating history"),
    ("review", "Review a past game from PGN"),
    ("quit", "Exit ChessCore"),
]

# Draw pieces as chess symbols (True) or letters (False). Changed by
# running "python -m chesscore --letters", for terminals that show the
# symbols as boxes (e.g. the old cmd window).
USE_SYMBOLS = True

# Draw the board with coloured light and dark squares (True) or as plain
# text (False). Turned off by "--plain", or when the terminal has no colours.
USE_COLORS = True

# ANSI escape codes (256-colour palette) for the coloured board.
RESET = "\033[0m"
LIGHT_SQUARE = "\033[48;5;180m"
DARK_SQUARE = "\033[48;5;137m"
LIGHT_HIGHLIGHT = "\033[48;5;186m"   # last move, on a light square
DARK_HIGHLIGHT = "\033[48;5;143m"    # last move, on a dark square
WHITE_PIECE = "\033[1;38;5;231m"
BLACK_PIECE = "\033[1;38;5;16m"
BOARD_SYMBOLS = {
    chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜",
    chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♙",
}
ANSI_CODE = re.compile(r"\033\[[0-9;]*m")

MOVES_PER_COLUMN = 20
HISTORY_COLUMN_WIDTH = 22


def piece_text(piece, symbols):
    """One character for a piece: a chess symbol, or a letter (uppercase White)."""
    if not symbols:
        return piece.symbol()
    # Terminals usually have a dark background, where the filled symbols
    # look light, so White gets the filled ones.
    return piece.unicode_symbol(invert_color=True)


def render_board(board, flipped=False, symbols=False):
    """Board as text lines, with letters (uppercase White) or chess symbols."""
    ranks = range(8) if flipped else range(7, -1, -1)
    files = range(7, -1, -1) if flipped else range(8)
    header = "  " + " ".join(chess.FILE_NAMES[f] for f in files)
    lines = [header]
    for rank in ranks:
        cells = []
        for file in files:
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece_text(piece, symbols) if piece else ".")
        lines.append(f"{rank + 1} " + " ".join(cells))
    lines.append(header)
    return lines


def render_color_board(board, flipped=False, symbols=False, last_move=None):
    """Board with coloured squares, three characters wide each.

    Every square is " X " (or "   " when empty), so the columns stay lined
    up whatever piece stands on them. The side of a piece is shown by its
    colour, so both sides use the same symbols: the filled ones, which show
    the colour best, except the pawn. The filled pawn is an emoji in some
    fonts and is drawn twice as wide, which breaks the grid.
    """
    ranks = range(8) if flipped else range(7, -1, -1)
    files = range(7, -1, -1) if flipped else range(8)
    highlighted = {last_move.from_square, last_move.to_square} if last_move else set()
    header = "  " + "".join(f" {chess.FILE_NAMES[f]} " for f in files)
    lines = [header]
    for rank in ranks:
        row = f"{rank + 1} "
        for file in files:
            square = chess.square(file, rank)
            light = (file + rank) % 2 == 1
            if square in highlighted:
                background = LIGHT_HIGHLIGHT if light else DARK_HIGHLIGHT
            else:
                background = LIGHT_SQUARE if light else DARK_SQUARE
            piece = board.piece_at(square)
            if piece is None:
                text = " "
            elif symbols:
                text = BOARD_SYMBOLS[piece.piece_type]
            else:
                text = piece.symbol()
            if piece is not None:
                text = (WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE) + text
            row += f"{background} {text} {RESET}"
        lines.append(row + f" {rank + 1}")
    lines.append(header)
    return lines


def visible_width(text):
    """Length of text on screen, ignoring colour codes."""
    return len(ANSI_CODE.sub("", text))


def render_history(sans):
    """Move history in SAN, one line per full move.

    After MOVES_PER_COLUMN full moves a new column starts to the right, so
    the history never grows taller than the board and pushes it off screen.
    """
    rows = []
    for i in range(0, len(sans), 2):
        white = sans[i]
        black = sans[i + 1] if i + 1 < len(sans) else "..."
        rows.append(f"{i // 2 + 1:>3}. {white:<9}{black:<8}")

    columns = [rows[i:i + MOVES_PER_COLUMN]
               for i in range(0, len(rows), MOVES_PER_COLUMN)] or [[]]
    width = len(columns) * (HISTORY_COLUMN_WIDTH + 2) - 2
    lines = ["MOVE HISTORY", "-" * width]
    for r in range(len(columns[0])):
        cells = [col[r] if r < len(col) else "" for col in columns]
        lines.append("  ".join(f"{c:<{HISTORY_COLUMN_WIDTH}}" for c in cells).rstrip())
    return lines


def format_score(score):
    if score is None:
        return "-"
    if abs(score) >= MATE_SCORE:
        return "mate" if score > 0 else "-mate"
    return f"{score / 100:+.1f}"


def render_screen(board, sans, flipped, last_score, depth, symbols=False,
                  colors=False):
    if colors:
        last_move = board.peek() if board.move_stack else None
        left = render_color_board(board, flipped, symbols, last_move)
    else:
        left = render_board(board, flipped, symbols)
    right = render_history(sans)
    height = max(len(left), len(right))
    left += [""] * (height - len(left))
    right += [""] * (height - len(right))
    width = max(visible_width(l) for l in left) + 4
    lines = [l + " " * (width - visible_width(l)) + r
             for l, r in zip(left, right)]
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
        move = board.parse_san(text)
    except ValueError:
        return None
    # parse_san turns inputs like "--" or "0000" into a null move (a passed
    # turn), which is not a real move in a game.
    return move if move else None


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
        print(render_screen(board, sans, human == chess.BLACK, last_score, depth,
                            USE_SYMBOLS, USE_COLORS))
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
    print(render_screen(board, sans, human == chess.BLACK, last_score, depth,
                            USE_SYMBOLS, USE_COLORS))
    print(game_result_text(board))


def ask_elo():
    while True:
        answer = input("external engine rating, empty for full strength > ").strip()
        if not answer:
            return None
        if answer.isdigit():
            return int(answer)
        print("Rating must be a whole number, e.g. 1500.")


def match():
    """Engine vs external engine (REQ-007-002): watch a whole game."""
    color = ask("ChessCore plays white or black? [w/b] > ", ["w", "b"])
    chesscore_color = chess.WHITE if color == "w" else chess.BLACK
    depth = ask_depth()
    elo = ask_elo()

    try:
        external = ExternalEngine(elo=elo)
    except FileNotFoundError as error:
        print(error)
        return

    sans = []

    def show(board, move):
        before = board.copy()
        before.pop()
        sans.append(before.san(move))
        print()
        print(render_screen(board, sans, False, None, depth, USE_SYMBOLS,
                            USE_COLORS))

    with external:
        white = "ChessCore" if chesscore_color == chess.WHITE else external.name
        black = external.name if chesscore_color == chess.WHITE else "ChessCore"
        print(f"{white} (white) vs {black} (black)")
        board = play_match_game(external, chesscore_color, depth, on_move=show)

    print(f"{white} vs {black}: {match_result(board)}")


def ask_number(prompt, default, minimum=1):
    while True:
        answer = input(f"{prompt} [{default}] > ").strip()
        if not answer:
            return default
        if answer.isdigit() and int(answer) >= minimum:
            return int(answer)
        print(f"Please enter a whole number of at least {minimum}.")


def train():
    """Training mode (REQ-012): games against Stockfish, then weight tuning."""
    games = ask_number("number of games", 10)
    depth = ask_depth()
    elo = ask_number("Stockfish rating", 1320, minimum=1)

    try:
        external = ExternalEngine(elo=elo)
    except FileNotFoundError as error:
        print(error)
        return

    storage = Storage()
    print(f"Training ChessCore v{storage.current_version()['version']} "
          f"against {external.name}: {games} games at depth {depth}")

    def show(number, game, score, stats):
        outcome = {1: "won", 0.5: "draw", 0: "lost"}[score]
        saved = "  (saved as PGN)" if "pgn" in game else ""
        print(f"game {number}/{games}: {game['white']} vs {game['black']} "
              f"{game['result']}, {outcome}, rating {stats['rating']:.0f}{saved}")

    with external:
        summary = run_training(storage, external, elo, games, depth, on_game=show)

    print()
    print(f"Results: {summary['wins']} won, {summary['draws']} drawn, "
          f"{summary['losses']} lost. Estimated rating {summary['rating']:.0f}.")
    if summary["error_before"] is None:
        print(f"Collected {summary['positions']} positions so far; tuning starts "
              f"at {MIN_POSITIONS} (about {MIN_POSITIONS // 50} games).")
        return
    if summary["new_version"] is None:
        print("Tuning found no better weights; the engine version stays the same.")
        return
    new = summary["new_version"]
    print(f"Tuned the evaluation on {summary['positions']} positions "
          f"(error {summary['error_before']:.4f} -> {summary['error_after']:.4f}).")
    print(f"New engine version v{new['version']}:")
    for name, value in new["weights"].items():
        old = summary["old_weights"][name]
        change = f"  ({value - old:+d})" if value != old else ""
        print(f"  {name:<9}{value}{change}")


def stats():
    """Training statistics and rating history per engine version."""
    storage = Storage()
    all_stats = storage.stats()
    if not all_stats:
        print("No training games yet. Choose 'train' to play some.")
        return
    print(f"{'version':<9}{'games':>6}{'won':>6}{'drawn':>7}{'lost':>6}{'rating':>8}")
    for version in storage.versions():
        number = version["version"]
        if str(number) not in all_stats:
            continue
        s = all_stats[str(number)]
        print(f"v{number:<8}{s['games']:>6}{s['wins']:>6}{s['draws']:>7}"
              f"{s['losses']:>6}{s['rating']:>8.0f}")
    current = storage.current_version()
    print(f"Current version: v{current['version']} ({current['note']})")


def print_menu():
    print()
    print("CHESSCORE")
    print(f"Terminal Chess Engine - v{VERSION} - 100% Python")
    print("-" * 60)
    for i, (name, description) in enumerate(MENU, start=1):
        print(f"[{i}] {name:<8} {description}")


def terminal_can_show_symbols():
    """False when the output encoding has no chess symbols (e.g. cp437)."""
    try:
        "♔♚".encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def enable_colors():
    """Turn on colour codes in the terminal. False if it cannot show them."""
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    # Windows consoles only understand colour codes after this mode is set.
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # standard output
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        return bool(kernel32.SetConsoleMode(
            handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
    except (AttributeError, OSError):
        return False


def main(argv=()):
    """Run the menu.

    argv: command-line options. "--letters" draws pieces as letters,
    "--plain" draws the board without colours.
    """
    global USE_SYMBOLS, USE_COLORS
    USE_SYMBOLS = "--letters" not in argv and terminal_can_show_symbols()
    USE_COLORS = "--plain" not in argv and enable_colors()
    # Play with the latest trained engine version.
    engine.set_weights(Storage().current_version()["weights"])

    names = [name for name, _ in MENU]
    while True:
        print_menu()
        choice = input("> ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            choice = names[int(choice) - 1]
        if choice == "play":
            play()
        elif choice == "match":
            match()
        elif choice == "train":
            train()
        elif choice == "stats":
            stats()
        elif choice == "quit":
            return
        elif choice in names:
            print(f"'{choice}' is not implemented yet.")
        else:
            print(f"Unknown option: {choice}")
