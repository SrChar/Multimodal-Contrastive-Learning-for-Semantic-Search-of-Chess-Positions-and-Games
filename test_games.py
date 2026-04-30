"""Streamlit app for text-to-game retrieval using a trained game-level ChessCLIP model."""

import ast
import os
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import streamlit as st
import chess
import chess.svg
from tqdm import tqdm

from train_games import ChessGamesCLIP

# ==========================
# Config
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GAME_CHECKPOINT_PATH = os.path.join(
    "checkpoints",
    "game_models",
    "chess_games_clip_best_0.1357.pt",
)

GAMES_CSV_PATH = os.path.join("datasets", "lichess_db_1M_streamlit.csv")

BATCH_SIZE_ENCODE_GAMES = 512      # number of games per batch for encoding
MAX_GAMES_TO_ENCODE: Optional[int] = 100000
MAX_MOVES_PER_GAME: Optional[int] = None

TOP_K_DEFAULT = 10

# Directory to store game embeddings + metadata
GAME_EMBEDDINGS_DIR = "game_embeddings"
GAME_EMBEDDINGS_FILE = os.path.join(GAME_EMBEDDINGS_DIR, "game_embeddings.npy")
GAME_METADATA_FILE = os.path.join(GAME_EMBEDDINGS_DIR, "games_metadata.csv")


# ==========================
# Helpers: board encoding
# ==========================

PIECE_TO_CHAN = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def board_to_planes(board: chess.Board) -> torch.Tensor:
    """Convert a python-chess Board into a (12, 8, 8) tensor."""
    import numpy as np

    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        symbol = piece.symbol()
        chan = PIECE_TO_CHAN.get(symbol)
        if chan is None:
            continue
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        row = 7 - rank
        col = file
        planes[chan, row, col] = 1.0

    return torch.from_numpy(planes)


def parse_moves_list(moves_raw) -> List[str]:
    """Parse the 'moves' column stored as a Python-list string or similar."""
    if isinstance(moves_raw, list):
        moves = [str(m) for m in moves_raw]
    elif isinstance(moves_raw, str):
        try:
            moves = ast.literal_eval(moves_raw)
        except Exception:
            # Fallback: split by comma
            moves = moves_raw.replace("[", "").replace("]", "").split(",")
        moves = [str(m).strip().strip('"').strip("'") for m in moves if str(m).strip()]
    else:
        moves = []

    return moves


# ==========================
# Build game tensors
# ==========================

def build_games_from_df(
    df: pd.DataFrame,
    max_games: Optional[int] = None,
    max_moves_per_game: Optional[int] = None,
) -> Tuple[List[torch.Tensor], pd.DataFrame]:
    """
    From a dataframe with columns [game_id, moves, num_moves], build:

      - a list of tensors, one per game: game_tensor of shape (T, 12, 8, 8)
      - a metadata dataframe with one row per game:
        [game_idx, game_id, num_plies, moves_str, num_moves (from CSV if present)]

    Notes:
    - Invalid moves inside a game cause that game to be skipped.
    - max_moves_per_game can be used to truncate very long games.
    """

    game_tensors: List[torch.Tensor] = []
    meta_rows = []

    # Rough estimate of total games for tqdm
    total_games = len(df)
    if max_games is not None:
        total_games = min(total_games, max_games)

    with tqdm(total=total_games, desc="Building game tensors", unit="games") as pbar:
        for _, row in df.iterrows():
            if max_games is not None and len(game_tensors) >= max_games:
                break

            game_id = row["game_id"]
            moves_raw = row["moves"]
            moves = parse_moves_list(moves_raw)

            if len(moves) == 0:
                continue

            if max_moves_per_game is not None:
                moves = moves[:max_moves_per_game]

            board = chess.Board()
            positions = []

            valid_game = True
            for ply_idx, san in enumerate(moves, start=1):
                try:
                    move = board.parse_san(san)
                except ValueError:
                    # Skip this game if an invalid move appears
                    print(
                        f"Warning: invalid SAN '{san}' in game {game_id}, "
                        f"ply {ply_idx}. Skipping this game."
                    )
                    valid_game = False
                    break

                board.push(move)
                planes = board_to_planes(board)  # (12, 8, 8)
                positions.append(planes)

            if not valid_game or len(positions) == 0:
                continue

            game_tensor = torch.stack(positions, dim=0)  # (T, 12, 8, 8)
            game_tensors.append(game_tensor)

            num_plies = game_tensor.shape[0]
            meta_rows.append(
                {
                    "game_idx": len(game_tensors) - 1,
                    "game_id": game_id,
                    "num_plies": num_plies,
                    "moves_str": str(moves_raw),  # original raw representation
                    "num_moves": row.get("num_moves", num_plies),
                }
            )

            pbar.update(1)

    if not game_tensors:
        raise RuntimeError("No valid games were built from the dataframe.")

    meta_df = pd.DataFrame(meta_rows)
    return game_tensors, meta_df


def pad_and_stack_games(game_tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of game tensors [ (T_i, 12, 8, 8), ... ] to a batch:

      batch_games: (B, T_max, 12, 8, 8)
      lengths:     (B,)

    This batch can be passed to model.encode_game.
    """
    lengths = torch.tensor([g.shape[0] for g in game_tensors], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(game_tensors)

    batch = torch.zeros((batch_size, max_len, 12, 8, 8), dtype=torch.float32)

    for i, g in enumerate(game_tensors):
        T = g.shape[0]
        batch[i, :T] = g

    return batch, lengths


# ==========================
# Model & indexing
# ==========================

@st.cache_resource
def load_model_and_game_index(
    checkpoint_path: str,
    games_csv_path: str,
    max_games: Optional[int],
    max_moves_per_game: Optional[int],
    device: str = DEVICE,
):
    """
    Load ChessGamesCLIP game-text model, build the index of game embeddings and metadata.

    Assumes ChessGamesCLIP implements:
        encode_game(game_batch: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor
    where:
        game_batch: (B, T, 12, 8, 8), lengths: (B,), output: (B, D) normalized.
    """

    model = ChessGamesCLIP(base_ckpt=None, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]

    # Load the checkpoint exactly as it was saved so encoders + poolers align
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # If embeddings already exist, load them
    if os.path.exists(GAME_EMBEDDINGS_DIR) and os.path.exists(GAME_EMBEDDINGS_FILE) and os.path.exists(
        GAME_METADATA_FILE
    ):
        print(f"Loading game embeddings from {GAME_EMBEDDINGS_DIR}...")
        game_embs_np = np.load(GAME_EMBEDDINGS_FILE, mmap_mode="r")
        meta_df = pd.read_csv(GAME_METADATA_FILE)
        return model, game_embs_np, meta_df

    # Otherwise, build from CSV
    games_df = pd.read_csv(games_csv_path)
    if not {"game_id", "moves"}.issubset(games_df.columns):
        raise ValueError("CSV must contain at least columns: ['game_id', 'moves'].")

    if "num_moves" not in games_df.columns:
        games_df["num_moves"] = games_df["moves"].apply(
            lambda x: len(parse_moves_list(x))
        )

    build_start = time.time()
    game_tensors, meta_df = build_games_from_df(
        games_df,
        max_games=max_games,
        max_moves_per_game=max_moves_per_game,
    )
    build_elapsed = time.time() - build_start
    build_speed = len(game_tensors) / build_elapsed if build_elapsed > 0 else 0.0
    print(
        f"Building game tensors completed in {build_elapsed:.2f} seconds "
        f"({build_speed:.2f} games/second)."
    )

    # Encode all games with the game encoder
    all_embs = []
    start_time = time.time()
    num_games = len(game_tensors)

    with torch.no_grad():
        for start in tqdm(
            range(0, num_games, BATCH_SIZE_ENCODE_GAMES),
            desc="Encoding games",
            unit="batch",
        ):
            end = min(start + BATCH_SIZE_ENCODE_GAMES, num_games)
            batch_games_list = game_tensors[start:end]
            batch_games, lengths = pad_and_stack_games(batch_games_list)

            batch_games = batch_games.to(device)
            lengths = lengths.to(device)

            emb = model.encode_game(batch_games, lengths)
            all_embs.append(emb.cpu())

    elapsed = time.time() - start_time
    speed = num_games / elapsed if elapsed > 0 else 0.0
    print(
        f"Encoding games completed in {elapsed:.2f} seconds "
        f"({speed:.2f} games/second)."
    )

    game_embs = torch.cat(all_embs, dim=0)
    game_embs_np = game_embs.numpy().astype("float32")

    # Save to disk
    os.makedirs(GAME_EMBEDDINGS_DIR, exist_ok=True)
    np.save(GAME_EMBEDDINGS_FILE, game_embs_np)
    meta_df.to_csv(GAME_METADATA_FILE, index=False)
    print(f"Saved game embeddings to {GAME_EMBEDDINGS_DIR}")

    return model, game_embs_np, meta_df


@torch.no_grad()
def encode_query(model: ChessGamesCLIP, query: str, device: str = DEVICE) -> np.ndarray:
    """Encode a text query into the same embedding space as games."""
    model.to(device)
    text_seqs = [[query]]  # mimic the training-time batching logic
    emb = model._encode_texts(text_seqs)
    emb = emb.to(device)
    return emb[0].cpu().numpy().astype("float32")


def search_top_k_games(
    query_emb: np.ndarray,
    game_embs: np.ndarray,
    meta_df: pd.DataFrame,
    top_k: int,
):
    """Return top-k matches (indices, scores, metadata rows) for games."""
    scores = game_embs @ query_emb  # cosine similarity if embeddings are normalized
    sorted_indices = np.argsort(-scores)  # descending

    selected_indices = []
    selected_scores = []

    for idx in sorted_indices:
        selected_indices.append(idx)
        selected_scores.append(scores[idx])
        if len(selected_indices) >= top_k:
            break

    results_meta = meta_df.iloc[selected_indices].copy()
    results_meta["score"] = selected_scores
    return selected_indices, selected_scores, results_meta


# ==========================
# HTML board rendering
# ==========================

def board_svg_from_board(board: chess.Board, size: int = 400) -> str:
    """Render a chess board as an SVG string."""
    return chess.svg.board(board=board, size=size)


# ==========================
# Streamlit app
# ==========================

def main():
    st.title("ChessCLIP – Text-to-game search")

    st.write(
        """
        This app loads the ChessCLIP **game-text** model and a set of games from CSV.
        It builds a database of full-game embeddings and lets you search them with a natural-language query.
        When you select a game, you can step through the moves one by one.
        """
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    ckpt_path = st.sidebar.text_input("Checkpoint path", value=GAME_CHECKPOINT_PATH)
    games_path = st.sidebar.text_input("Games CSV path", value=GAMES_CSV_PATH)

    top_k = st.sidebar.slider(
        "Top-K results",
        min_value=1,
        max_value=50,
        value=TOP_K_DEFAULT,
        step=1,
    )

    max_games_to_encode = st.sidebar.number_input(
        "Max games to encode (None = all)",
        min_value=1,
        value=MAX_GAMES_TO_ENCODE if MAX_GAMES_TO_ENCODE is not None else 10000,
        step=1000,
    )
    use_max_games_limit = st.sidebar.checkbox("Use max games limit?", value=(MAX_GAMES_TO_ENCODE is not None))

    max_moves_per_game = st.sidebar.number_input(
        "Max moves per game (0 = no limit)",
        min_value=0,
        value=MAX_MOVES_PER_GAME if MAX_MOVES_PER_GAME is not None else 0,
        step=20,
    )

    # Convert 0 / checkbox into proper None / int
    max_games_arg: Optional[int] = None
    if use_max_games_limit:
        max_games_arg = int(max_games_to_encode)

    max_moves_arg: Optional[int] = None
    if max_moves_per_game > 0:
        max_moves_arg = int(max_moves_per_game)

    # Load model and index
    try:
        with st.spinner("Loading model and building game index..."):
            model, game_embs, meta_df = load_model_and_game_index(
                checkpoint_path=ckpt_path,
                games_csv_path=games_path,
                max_games=max_games_arg,
                max_moves_per_game=max_moves_arg,
                device=DEVICE,
            )
    except Exception as e:
        st.error(f"Error loading model/index: {e}")
        return

    st.success(
        f"Index ready: {game_embs.shape[0]} games encoded."
    )

    # Query input
    query = st.text_input(
        "Write a chess game description (in English or Spanish):",
        "",
    )

    if query:
        query_emb = encode_query(model, query, device=DEVICE)
        _, _, results_meta = search_top_k_games(
            query_emb,
            game_embs,
            meta_df,
            top_k=top_k,
        )

        st.subheader("Top results")

        option_labels = []
        label_to_idx = {}

        for rank, (_, row) in enumerate(results_meta.iterrows(), start=1):
            label = (
                f"{rank}. Game {row['game_id']} – plies {row['num_plies']} "
                f"(score={row['score']:.3f})"
            )
            option_labels.append(label)

            # Map label to the index in meta_df
            label_to_idx[label] = row.name

        chosen_label = st.selectbox(
            "Select a game to visualize:",
            options=option_labels,
        )

        if chosen_label:
            idx = label_to_idx[chosen_label]
            row = meta_df.loc[idx]

            st.markdown(f"**Selected game:** `{row['game_id']}`")
            st.markdown(f"Number of plies stored: `{int(row['num_plies'])}`")

            # Parse moves and build boards for stepping through the game
            moves = parse_moves_list(row["moves_str"])
            if len(moves) == 0:
                st.error("This game has no parsable moves.")
                return

            # Slider to choose ply
            max_ply = len(moves)
            ply = st.slider(
                "Move index (ply, half-move)",
                min_value=0,
                max_value=max_ply,
                value=0,
                step=1,
            )

            # Rebuild board up to selected ply
            board = chess.Board()
            last_san = None
            if ply > 0:
                for i in range(ply):
                    san = moves[i]
                    try:
                        move = board.parse_san(san)
                    except ValueError:
                        st.error(f"Invalid SAN '{san}' at ply {i+1}.")
                        break
                    board.push(move)
                    last_san = san

            svg = board_svg_from_board(board)
            st.components.v1.html(svg, height=420, width=420)

            st.json(
                {
                    "game_id": str(row["game_id"]),
                    "current_ply": int(ply),
                    "last_move_san": last_san if last_san is not None else None,
                    "fen": board.fen(),
                    "total_plies": int(len(moves)),
                }
            )

            with st.expander("Show full move list"):
                st.write(" ".join(moves))


if __name__ == "__main__":
    main()
