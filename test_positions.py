"""Streamlit app for text-to-position retrieval using a trained position-level ChessCLIP model."""

import ast
import os
import time
from typing import List

import numpy as np
import pandas as pd
import torch
import streamlit as st
import chess
import chess.svg
from tqdm import tqdm
from train_positions import ChessCLIP, TEXT_MODEL_NAME, PROJ_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = os.path.join("checkpoints", "position_model/chess_clip_best_0.2481.pt")
GAMES_CSV_PATH = os.path.join("datasets/", "lichess_db_1M_streamlit.csv")
BATCH_SIZE_ENCODE = 256
TOP_K_DEFAULT = 10
MAX_POSITIONS_TO_ENCODE = 100_000_000
UNIQUE_GAMES_ONLY = True

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
    """Parse the 'moves' column which is stored as a Python-list string."""
    if isinstance(moves_raw, list):
        moves = [str(m) for m in moves_raw]
    elif isinstance(moves_raw, str):
        try:
            moves = ast.literal_eval(moves_raw)
        except Exception:
            moves = moves_raw.replace("[", "").replace("]", "").split(",")
        moves = [str(m).strip().strip('"').strip("'") for m in moves if str(m).strip()]
    else:
        moves = []

    return moves


def position_batches_from_games(df: pd.DataFrame, limit: int = None, batch_size: int = 512):
    """Yield batches of board planes and metadata without holding everything in memory."""
    positions: List[torch.Tensor] = []
    meta_rows: List[dict] = []
    produced = 0

    total_est = int(df["num_moves"].sum()) if "num_moves" in df.columns else len(df) * 40
    total_for_tqdm = min(limit, total_est) if limit is not None else total_est

    with tqdm(total=total_for_tqdm, desc="Building positions", unit="positions") as pbar:
        for _, row in df.iterrows():
            if limit is not None and produced >= limit:
                break
            game_id = row["game_id"]
            moves = parse_moves_list(row["moves"])

            board = chess.Board()
            for ply_idx, san in enumerate(moves, start=1):
                if limit is not None and produced >= limit:
                    break
                try:
                    move = board.parse_san(san)
                except ValueError:
                    print(f"Warning: invalid SAN '{san}' in game {game_id}, ply {ply_idx}. Stopping this game.")
                    break

                board.push(move)
                positions.append(board_to_planes(board))
                meta_rows.append(
                    {
                        "game_id": game_id,
                        "ply_idx": ply_idx,
                        "san": san,
                        "fen": board.fen(),
                    }
                )
                produced += 1
                pbar.update(1)

                if len(positions) >= batch_size:
                    yield positions, meta_rows
                    positions, meta_rows = [], []

    if positions:
        yield positions, meta_rows


@st.cache_resource
def load_model_and_index(
    checkpoint_path: str,
    games_csv_path: str,
    limit: int = None,
    device: str = DEVICE,
):
    """Load ChessCLIP, build the index of board embeddings and metadata."""

    model = ChessCLIP(text_model_name=TEXT_MODEL_NAME, proj_dim=PROJ_DIM, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    EMBEDDINGS_DIR = "positions_embeddings"
    EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "board_embeddings.npy")
    METADATA_FILE = os.path.join(EMBEDDINGS_DIR, "metadata.csv")

    if os.path.exists(EMBEDDINGS_DIR) and os.listdir(EMBEDDINGS_DIR):
        print(f"Loading embeddings from {EMBEDDINGS_DIR}...")
        meta_df = pd.read_csv(METADATA_FILE)
        num_positions = len(meta_df)
        board_embs_np = np.memmap(
            EMBEDDINGS_FILE,
            mode="r",
            dtype="float32",
            shape=(num_positions, PROJ_DIM),
        )
        return model, board_embs_np, meta_df

    games_df = pd.read_csv(games_csv_path)
    if not {"game_id", "moves", "num_moves"}.issubset(games_df.columns):
        raise ValueError("CSV must contain columns: ['game_id', 'moves', 'num_moves'].")

    total_est = int(games_df["num_moves"].sum()) if "num_moves" in games_df.columns else len(games_df) * 40
    if limit is not None:
        total_est = min(limit, total_est)

    if total_est <= 0:
        raise RuntimeError("No positions to encode: num_moves sum is zero or limit is zero.")

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    board_embs_memmap = np.memmap(
        EMBEDDINGS_FILE,
        mode="w+",
        dtype="float32",
        shape=(total_est, PROJ_DIM),
    )

    meta_rows = []
    write_ptr = 0

    start_time = time.time()
    with torch.no_grad():
        for batch_positions, batch_meta in position_batches_from_games(
            games_df,
            limit=limit,
            batch_size=BATCH_SIZE_ENCODE,
        ):
            batch_tensor = torch.stack(batch_positions, dim=0).to(device)
            emb = model.encode_board(batch_tensor).cpu().numpy().astype("float32")

            end_ptr = write_ptr + emb.shape[0]
            board_embs_memmap[write_ptr:end_ptr] = emb

            for i, meta in enumerate(batch_meta):
                meta_rows.append({
                    "pos_id": write_ptr + i,
                    **meta,
                })

            write_ptr = end_ptr

            del batch_tensor, emb
            if device == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    speed = write_ptr / elapsed if elapsed > 0 else 0
    print(f"Encoding completed in {elapsed:.2f} seconds. Speed: {speed:.2f} positions/second. Encoded {write_ptr} positions.")

    board_embs_memmap.flush()

    # Save metadata only for positions that were actually written.
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(METADATA_FILE, index=False)

    # Re-open the file using the real shape (write_ptr) to avoid trailing garbage.
    board_embs_np = np.memmap(
        EMBEDDINGS_FILE,
        mode="r",
        dtype="float32",
        shape=(write_ptr, PROJ_DIM),
    )

    print(f"Saved embeddings to {EMBEDDINGS_DIR} with {write_ptr} positions.")

    return model, board_embs_np, meta_df


@torch.no_grad()
def encode_query(model: ChessCLIP, query: str, device: str = DEVICE) -> np.ndarray:
    """Encode a text query into the same embedding space."""
    model.to(device)
    text_emb = model.encode_text([query])
    return text_emb[0].cpu().numpy().astype("float32")


def search_top_k(
    query_emb: np.ndarray,
    board_embs: np.ndarray,
    meta_df: pd.DataFrame,
    top_k: int,
    unique_games: bool = False,
    unique_positions: bool = False,
):
    """Return top-k matches (indices, scores, metadata rows)."""
    scores = board_embs @ query_emb
    sorted_indices = np.argsort(-scores)

    selected_indices = []
    selected_scores = []
    seen_games = set()
    seen_positions = set()

    for idx in sorted_indices:
        game_id = meta_df.iloc[idx]["game_id"]
        fen = meta_df.iloc[idx]["fen"]
        if unique_games and game_id in seen_games:
            continue
        if unique_positions and fen in seen_positions:
            continue
        selected_indices.append(idx)
        selected_scores.append(scores[idx])
        if unique_games:
            seen_games.add(game_id)
        if unique_positions:
            seen_positions.add(fen)
        if len(selected_indices) >= top_k:
            break

    results_meta = meta_df.iloc[selected_indices].copy()
    results_meta["score"] = selected_scores
    return selected_indices, selected_scores, results_meta


def board_svg_from_fen(fen: str, size: int = 400) -> str:
    """Render a chess board from FEN as an SVG string using chess.svg."""
    board = chess.Board(fen)
    return chess.svg.board(board=board, size=size)


def main():
    """Render the Streamlit UI for querying positions with natural language."""
    st.title("ChessDualCLIP – Text-to-position search")

    st.write(
        """
        This app loads your trained ChessCLIP model and a set of games from CSV.
        It builds a database of all positions and lets you search them with a natural-language query.
        """
    )

    st.sidebar.header("Configuration")
    ckpt_path = st.sidebar.text_input("Checkpoint path", value=CHECKPOINT_PATH)
    games_path = st.sidebar.text_input("Games CSV path", value=GAMES_CSV_PATH)
    top_k = st.sidebar.slider("Top-K results", min_value=1, max_value=50, value=TOP_K_DEFAULT, step=1)
    limit_positions = st.sidebar.number_input("Max positions to encode", min_value=100, value=MAX_POSITIONS_TO_ENCODE, step=100)
    unique_games = st.sidebar.checkbox("Unique games only", value=UNIQUE_GAMES_ONLY)
    unique_positions = st.sidebar.checkbox("Unique positions only", value=True)

    try:
        with st.spinner("Loading model and building index..."):
            model, board_embs, meta_df = load_model_and_index(
                checkpoint_path=ckpt_path,
                games_csv_path=games_path,
                limit=limit_positions,
                device=DEVICE,
            )
    except Exception as e:
        st.error(f"Error loading model/index: {e}")
        return

    st.success(
        f"Index ready: {board_embs.shape[0]} positions loaded from {len(meta_df['game_id'].unique())} games."
    )

    query = st.text_input("Write a chess position description (in English or Spanish):", "")

    if query:
        query_emb = encode_query(model, query, device=DEVICE)
        _, _, results_meta = search_top_k(query_emb, board_embs, meta_df, top_k=top_k, unique_games=unique_games, unique_positions=unique_positions)

        st.subheader("Top results")
        option_labels = []
        label_to_idx = {}

        for rank, (_, row) in enumerate(results_meta.iterrows(), start=1):
            label = (
                f"{rank}. Game {row['game_id']} – ply {row['ply_idx']} – move {row['san']} "
                f"(score={row['score']:.3f})"
            )
            option_labels.append(label)
            label_to_idx[label] = row.name

        chosen_label = st.selectbox("Select a position to visualize:", options=option_labels)

        if chosen_label:
            idx = label_to_idx[chosen_label]
            fen = meta_df.loc[idx, "fen"]
            svg = board_svg_from_fen(fen)
            st.components.v1.html(svg, height=420, width=420)

            st.json(
                {
                    "game_id": int(meta_df.loc[idx, "game_id"]),
                    "ply_idx": int(meta_df.loc[idx, "ply_idx"]),
                    "san": str(meta_df.loc[idx, "san"]),
                    "fen": str(meta_df.loc[idx, "fen"]),
                    "score": float(results_meta.loc[idx, "score"])
                    if idx in results_meta.index
                    else None,
                }
            )


if __name__ == "__main__":
    main()
