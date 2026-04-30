"""Convert a PGN file into a CSV dataset with SAN move sequences."""

import os
import sys
import pandas as pd
from tqdm import tqdm

try:
    import chess
    import chess.pgn
except Exception:
    print("ERROR: The 'python-chess' package is required. Install it with:\n  pip install python-chess", file=sys.stderr)
    raise

# Configuration
INPUT_PGN = os.path.join("datasets", "lichess_db_1M_streamlit.pgn")
OUTPUT_CSV = os.path.join("datasets", "lichess_db_1M_streamlit.csv")
MIN_MOVES = 8          # minimum SAN moves to accept a game
WRITE_CHUNK_SIZE = 1_000_000  # write every N accepted games

MAX_GAMES = None        # None = all possible; or set an integer to limit

def extract_moves_from_game(game: chess.pgn.Game) -> list[str]:
    """Extract SAN moves from the mainline of a PGN game."""
    board = game.board()
    moves_san = []
    for move in game.mainline_moves():
        san = board.san(move)  # SAN before push
        board.push(move)
        moves_san.append(san)
    return moves_san

def _flush_rows_to_csv(rows: list[dict[str, object]], output_csv: str, write_header: bool) -> None:
    """Write a chunk of rows to CSV and clear memory pressure."""
    if not rows:
        return

    df_chunk = pd.DataFrame(rows, columns=["game_id", "moves", "num_moves"])
    df_chunk.to_csv(
        output_csv,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        encoding="utf-8",
    )


def build_dataset_from_pgn(
    input_pgn: str,
    output_csv: str,
    min_moves: int = 46,
    max_games: int | None = None,
    write_chunk_size: int = 1_000_000,
) -> None:
    """Build a CSV dataset from PGN games filtered by minimum move count."""
    if not os.path.exists(input_pgn):
        raise FileNotFoundError(f"PGN not found: {input_pgn}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    accepted_rows = []
    accepted = 0
    seen = 0
    written = 0
    write_header = True

    # Start with a clean output file for this run.
    if os.path.exists(output_csv):
        os.remove(output_csv)

    with open(input_pgn, "r", encoding="utf-8", errors="ignore") as f:
        pbar = tqdm(desc=f"Processing PGN (>= {min_moves} moves)", unit="game")
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            seen += 1
            moves = extract_moves_from_game(game)

            if len(moves) >= min_moves:
                row = {
                    "game_id": accepted,      # sequential id for accepted games
                    "moves": moves,           # list of SAN moves
                    "num_moves": len(moves),  # number of moves
                }
                accepted_rows.append(row)
                accepted += 1

                if len(accepted_rows) >= write_chunk_size:
                    _flush_rows_to_csv(accepted_rows, output_csv, write_header)
                    written += len(accepted_rows)
                    accepted_rows.clear()
                    write_header = False
                    print(f"Flushed chunk to CSV. Written accepted games so far: {written}")

                if max_games is not None and accepted >= max_games:
                    pbar.update(1)
                    break

            pbar.update(1)
        pbar.close()

    print(f"Accepted games (>= {min_moves} moves): {accepted} out of {seen} read.")

    # Flush remaining rows.
    _flush_rows_to_csv(accepted_rows, output_csv, write_header)
    written += len(accepted_rows)
    accepted_rows.clear()

    print(f"Total written rows: {written}")
    print(f"CSV saved at: {output_csv}")

if __name__ == "__main__":
    build_dataset_from_pgn(
        INPUT_PGN,
        OUTPUT_CSV,
        min_moves=MIN_MOVES,
        max_games=MAX_GAMES,
        write_chunk_size=WRITE_CHUNK_SIZE,
    )