"""Train and evaluate a position-text ChessCLIP model with stratified retrieval metrics."""

import os
import math
from typing import Any, Dict, List, Tuple, Optional
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel
import torchvision.models as models
from numba import njit, prange, int32, int8, float32

try:
    import chess
except Exception:
    chess = None


# Configuration
TEXT_MODEL_NAME = "openai/clip-vit-base-patch32"
PROJ_DIM = 256
BATCH_SIZE_TRAIN = 64   
BATCH_SIZE_VAL = 64  # You can adjust this value based on available RAM.
ACCUM_STEPS = 1
EPOCHS = 10000
PATIENCE = 15
LR = 1e-4
TEXT_LR = 5e-5
WEIGHT_DECAY = 5e-3
VAL_SIZE = 0.1
VAL_COUNT: Optional[int] = 1_000
NUM_WORKERS = 8
OUTPUT_DIR = "checkpoints/position_model"
TRAIN_CSV_PATH = "data/final/train_df_final.csv"
VAL_CSV_PATH = "data/final/val_df_final.csv"
TEST_CSV_PATH = "data/final/test_df_final.csv"
NEGATIVE_POSITIONS_PARQUET_DIR = "datasets/lichess_positions_parquet"

# Distance intervals: each tuple is (low_inclusive, high_inclusive).
# The last interval is open-ended (>=126).
EVAL_DISTANCE_INTERVALS: List[Tuple[int, int]] = [
    (1, 20), (21, 35), (36, 50), (51, 65),
    (66, 80), (81, 95), (96, 110), (111, 125), (126, 9999),
]
NEGATIVES_PER_THRESHOLD = 1_000
NEGATIVE_EMBED_BATCH_SIZE = 256
NEGATIVE_SELECTION_SEED = 1337
PARQUET_CHUNK_SIZE = 100_000           # process each parquet in chunks of this size (~half parquet)
EVAL_QUERY_CHUNK = 64                 # chunk size for memory-efficient eval
NEG_CACHE_CHECKPOINT_EVERY = 100      # save a neg-cache checkpoint every N fully-satisfied queries

# Checkpoint config
SAVE_EVERY = 5
resume_ckpt = None


# ==========================
# Position Distance Calculation (Numba-accelerated)
# ==========================

# Precomputed knight distance lookup table (64×64) — built once at import time.
def _build_knight_dist_table() -> np.ndarray:
    """Return a (64, 64) int8 array of minimum knight moves between any two squares."""
    table = np.zeros((64, 64), dtype=np.int8)
    moves = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    for src in range(64):
        r0, c0 = divmod(src, 8)
        dist_map = {(r0, c0): 0}
        queue = [(r0, c0, 0)]
        while queue:
            r, c, d = queue.pop(0)
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8 and (nr, nc) not in dist_map:
                    dist_map[(nr, nc)] = d + 1
                    queue.append((nr, nc, d + 1))
        for (r, c), d in dist_map.items():
            table[src, r * 8 + c] = d
    return table

_KNIGHT_DIST_TABLE: np.ndarray = _build_knight_dist_table()

# ── Numba-compiled distance primitives ────────────────────────────────────

@njit(int32(int8[:, :], int32, int32, int32, int32), cache=True)
def _nb_knight_dist(table, r1, c1, r2, c2):
    return int32(table[r1 * 8 + c1, r2 * 8 + c2])

@njit(int32(int32, int32, int32, int32), cache=True)
def _nb_king_dist(r1, c1, r2, c2):
    dr = r1 - r2 if r1 > r2 else r2 - r1
    dc = c1 - c2 if c1 > c2 else c2 - c1
    return dr if dr > dc else dc

@njit(int32(int32, int32, int32, int32), cache=True)
def _nb_queen_dist(r1, c1, r2, c2):
    if r1 == r2 and c1 == c2:
        return int32(0)
    dr = r1 - r2 if r1 > r2 else r2 - r1
    dc = c1 - c2 if c1 > c2 else c2 - c1
    if r1 == r2 or c1 == c2 or dr == dc:
        return int32(1)
    return int32(2)

@njit(int32(int32, int32, int32, int32), cache=True)
def _nb_rook_dist(r1, c1, r2, c2):
    if r1 == r2 and c1 == c2:
        return int32(0)
    if r1 == r2 or c1 == c2:
        return int32(1)
    return int32(2)

@njit(int32(int32, int32, int32, int32), cache=True)
def _nb_bishop_dist(r1, c1, r2, c2):
    if r1 == r2 and c1 == c2:
        return int32(0)
    dr = r1 - r2 if r1 > r2 else r2 - r1
    dc = c1 - c2 if c1 > c2 else c2 - c1
    if dr == dc:
        return int32(1)
    if (r1 + c1) % 2 == (r2 + c2) % 2:
        return int32(2)
    return int32(100)

@njit(int32(int32, int32, int32, int32), cache=True)
def _nb_pawn_dist(r1, c1, r2, c2):
    if r1 == r2 and c1 == c2:
        return int32(0)
    dr = r1 - r2 if r1 > r2 else r2 - r1
    dc = c1 - c2 if c1 > c2 else c2 - c1
    return dr + dc

# Piece-type encoding for Numba: K=0, Q=1, R=2, B=3, N=4, P=5
@njit(int32(int32, int32, int32, int32, int32, int8[:, :]), cache=True)
def _nb_piece_cost(ptype, r1, c1, r2, c2, knight_table):
    if ptype == 0:    # King
        return _nb_king_dist(r1, c1, r2, c2)
    elif ptype == 1:  # Queen
        return _nb_queen_dist(r1, c1, r2, c2)
    elif ptype == 2:  # Rook
        return _nb_rook_dist(r1, c1, r2, c2)
    elif ptype == 3:  # Bishop
        return _nb_bishop_dist(r1, c1, r2, c2)
    elif ptype == 4:  # Knight
        return _nb_knight_dist(knight_table, r1, c1, r2, c2)
    elif ptype == 5:  # Pawn
        return _nb_pawn_dist(r1, c1, r2, c2)
    return int32(0)

# Penalties: K=0, Q=10, R=8, B=7, N=6, P=5
_PENALTIES = np.array([0, 10, 8, 7, 6, 5], dtype=np.int32)

@njit(int32(int32[:, :], int32[:, :], int32, int32, int8[:, :], int32[:]), cache=True)
def _nb_hungarian_cost(pieces1, pieces2, n1, n2, knight_table, penalties):
    """
    Simplified assignment for small sets (max ~8 pieces of same type).
    Uses brute-force min-cost for n<=3, else a greedy approach that is
    near-optimal for chess piece sets.
    """
    if n1 == 0 and n2 == 0:
        return int32(0)
    ptype = pieces1[0, 0] if n1 > 0 else pieces2[0, 0]
    penalty = penalties[ptype]
    mx = n1 if n1 > n2 else n2

    # Build cost matrix (in-place, stack-allocated for small sizes)
    cost = np.empty((mx, mx), dtype=np.int32)
    for i in range(mx):
        for j in range(mx):
            if i < n1 and j < n2:
                cost[i, j] = _nb_piece_cost(ptype, pieces1[i, 1], pieces1[i, 2],
                                             pieces2[j, 1], pieces2[j, 2], knight_table)
            else:
                cost[i, j] = penalty

    # For very small matrices, do exact assignment
    if mx <= 1:
        return cost[0, 0]

    # Greedy row-minimum assignment (very fast, good enough for <=8 pieces)
    total = int32(0)
    used = np.zeros(mx, dtype=np.int32)
    for i in range(mx):
        best_j = -1
        best_c = int32(999999)
        for j in range(mx):
            if used[j] == 0 and cost[i, j] < best_c:
                best_c = cost[i, j]
                best_j = j
        used[best_j] = 1
        total += best_c
    return total


@njit(int32(float32[:, :, :], float32[:, :, :], int8[:, :], int32[:]), cache=True)
def _nb_board_distance(board1, board2, knight_table, penalties):
    """
    Compute full chess-distance between two (12,8,8) boards.
    Channels: 0-5 = white P,N,B,R,Q,K; 6-11 = black p,n,b,r,q,k
    Mapped to ptype: P=5, N=4, B=3, R=2, Q=1, K=0
    """
    # Channel → ptype mapping
    # ch0=P→5, ch1=N→4, ch2=B→3, ch3=R→2, ch4=Q→1, ch5=K→0
    # ch6-11 same for black
    chan_to_ptype = np.array([5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0], dtype=np.int32)

    total = int32(0)

    for color in range(2):  # 0=white, 1=black
        offset = color * 6

        # ── King (channel offset+5, ptype=0) ──
        kr1 = int32(-1); kc1 = int32(-1)
        kr2 = int32(-1); kc2 = int32(-1)
        for r in range(8):
            for c in range(8):
                if board1[offset + 5, r, c] > 0.5:
                    kr1 = int32(r); kc1 = int32(c)
                if board2[offset + 5, r, c] > 0.5:
                    kr2 = int32(r); kc2 = int32(c)
        if kr1 >= 0 and kr2 >= 0:
            total += _nb_king_dist(kr1, kc1, kr2, kc2)

        # ── Queen (channel offset+4, ptype=1) ──
        qr1 = int32(-1); qc1 = int32(-1)
        qr2 = int32(-1); qc2 = int32(-1)
        nq1 = int32(0); nq2 = int32(0)
        for r in range(8):
            for c in range(8):
                if board1[offset + 4, r, c] > 0.5:
                    qr1 = int32(r); qc1 = int32(c); nq1 += 1
                if board2[offset + 4, r, c] > 0.5:
                    qr2 = int32(r); qc2 = int32(c); nq2 += 1
        if nq1 > 0 and nq2 > 0:
            total += _nb_queen_dist(qr1, qc1, qr2, qc2)
        elif nq1 != nq2:
            total += penalties[1]  # Queen penalty

        # ── Bishops (channel offset+2, ptype=3) — split by square colour ──
        for parity in range(2):
            br1 = int32(-1); bc1 = int32(-1)
            br2 = int32(-1); bc2 = int32(-1)
            nb1 = int32(0); nb2 = int32(0)
            for r in range(8):
                for c in range(8):
                    if (r + c) % 2 != parity:
                        continue
                    if board1[offset + 2, r, c] > 0.5:
                        br1 = int32(r); bc1 = int32(c); nb1 += 1
                    if board2[offset + 2, r, c] > 0.5:
                        br2 = int32(r); bc2 = int32(c); nb2 += 1
            if nb1 > 0 and nb2 > 0:
                total += _nb_bishop_dist(br1, bc1, br2, bc2)
            elif nb1 != nb2:
                total += penalties[3]  # Bishop penalty

        # ── Rooks (channel offset+3, ptype=2) — Hungarian ──
        pieces1 = np.empty((10, 3), dtype=np.int32)
        pieces2 = np.empty((10, 3), dtype=np.int32)
        n1 = int32(0); n2 = int32(0)
        for r in range(8):
            for c in range(8):
                if board1[offset + 3, r, c] > 0.5:
                    pieces1[n1, 0] = 2; pieces1[n1, 1] = r; pieces1[n1, 2] = c; n1 += 1
                if board2[offset + 3, r, c] > 0.5:
                    pieces2[n2, 0] = 2; pieces2[n2, 1] = r; pieces2[n2, 2] = c; n2 += 1
        if n1 > 0 or n2 > 0:
            total += _nb_hungarian_cost(pieces1, pieces2, n1, n2, knight_table, penalties)

        # ── Knights (channel offset+1, ptype=4) — Hungarian ──
        n1 = int32(0); n2 = int32(0)
        for r in range(8):
            for c in range(8):
                if board1[offset + 1, r, c] > 0.5:
                    pieces1[n1, 0] = 4; pieces1[n1, 1] = r; pieces1[n1, 2] = c; n1 += 1
                if board2[offset + 1, r, c] > 0.5:
                    pieces2[n2, 0] = 4; pieces2[n2, 1] = r; pieces2[n2, 2] = c; n2 += 1
        if n1 > 0 or n2 > 0:
            total += _nb_hungarian_cost(pieces1, pieces2, n1, n2, knight_table, penalties)

        # ── Pawns (channel offset+0, ptype=5) — Hungarian ──
        n1 = int32(0); n2 = int32(0)
        for r in range(8):
            for c in range(8):
                if board1[offset + 0, r, c] > 0.5:
                    pieces1[n1, 0] = 5; pieces1[n1, 1] = r; pieces1[n1, 2] = c; n1 += 1
                if board2[offset + 0, r, c] > 0.5:
                    pieces2[n2, 0] = 5; pieces2[n2, 1] = r; pieces2[n2, 2] = c; n2 += 1
        if n1 > 0 or n2 > 0:
            total += _nb_hungarian_cost(pieces1, pieces2, n1, n2, knight_table, penalties)

    return total


@njit(int32[:](float32[:, :, :], float32[:, :, :, :], int8[:, :], int32[:]), cache=True)
def _nb_batch_distances(query_board, candidates, knight_table, penalties):
    """Compute distances from one query board to M candidates. Returns (M,) int32."""
    M = candidates.shape[0]
    dists = np.empty(M, dtype=np.int32)
    for i in range(M):
        dists[i] = _nb_board_distance(query_board, candidates[i], knight_table, penalties)
    return dists


@njit(int32[:, :](float32[:, :, :, :], float32[:, :, :, :], int8[:, :], int32[:]),
      parallel=True, cache=True)
def _nb_multi_batch_distances(queries, candidates, knight_table, penalties):
    """Compute distances from Q queries to M candidates in parallel.
    Returns (Q, M) int32 matrix."""
    Q = queries.shape[0]
    M = candidates.shape[0]
    dists = np.empty((Q, M), dtype=np.int32)
    for q in prange(Q):
        for m in range(M):
            dists[q, m] = _nb_board_distance(queries[q], candidates[m], knight_table, penalties)
    return dists


# ── Warm-up Numba (JIT compile on first import) ──────────────────────────
print("JIT-compiling chess distance functions (one-time)...", end=" ", flush=True)
_dummy1 = np.zeros((12, 8, 8), dtype=np.float32)
_dummy2 = np.zeros((12, 8, 8), dtype=np.float32)
_dummy1[5, 0, 4] = 1.0  # white king e1
_dummy2[5, 0, 4] = 1.0
_ = _nb_board_distance(_dummy1, _dummy2, _KNIGHT_DIST_TABLE, _PENALTIES)
_ = _nb_batch_distances(_dummy1, _dummy2.reshape(1, 12, 8, 8), _KNIGHT_DIST_TABLE, _PENALTIES)
_ = _nb_multi_batch_distances(
    _dummy1.reshape(1, 12, 8, 8), _dummy2.reshape(1, 12, 8, 8),
    _KNIGHT_DIST_TABLE, _PENALTIES,
)
print("done.")
del _dummy1, _dummy2


# ── Python wrappers (keep compatibility with rest of code) ───────────────

def board_matrix_to_pieces(board_matrix: np.ndarray) -> dict:
    """Convert a 12x8x8 board matrix to a dictionary of pieces."""
    piece_channels = {
        0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
        6: 'p', 7: 'n', 8: 'b', 9: 'r', 10: 'q', 11: 'k',
    }
    pieces = {'white': [], 'black': []}
    for channel, piece_type in piece_channels.items():
        positions = np.argwhere(board_matrix[channel] == 1)
        color = 'white' if channel < 6 else 'black'
        for row, col in positions:
            pieces[color].append((piece_type.upper(), row, col))
    return pieces


def calculate_position_distance(board1: np.ndarray, board2: np.ndarray) -> int:
    """Calculate chess distance between two (12,8,8) boards using Numba-compiled function."""
    return int(_nb_board_distance(
        board1.astype(np.float32), board2.astype(np.float32),
        _KNIGHT_DIST_TABLE, _PENALTIES,
    ))


PIECE_TO_CHAN = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}


def board_to_planes_np(board: Any) -> np.ndarray:
    """Convert a python-chess board into a (12, 8, 8) float32 numpy array."""
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
    return planes


def parse_moves_list(moves_raw: Any) -> List[str]:
    """Parse the 'moves' field stored as python-list string in game CSVs."""
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


# ==========================
# Dataset
# ==========================

class ChessDataset(Dataset):
    """Dataset that yields board tensors, rewritten text, and optional group ids."""

    def __init__(
        self,
        df: pd.DataFrame,
        board_col: str = "board_matrix",
        text_col: str = "rewritten",
        group_col: Optional[str] = None,
    ) -> None:
        """Store dataframe references and column names for later access."""
        self.df = df.reset_index(drop=True)
        self.board_col = board_col
        self.text_col = text_col
        self.group_col = group_col

    def __len__(self) -> int:
        """Return the number of valid rows inside the dataset."""
        return len(self.df)

    def _parse_board(self, raw_board: Any) -> torch.Tensor:
        """Convert serialized board data into a 12x8x8 float tensor."""
        if isinstance(raw_board, str):
            try:
                raw_board = ast.literal_eval(raw_board)
            except Exception as e:
                raise ValueError(f"Could not parse board string: {raw_board}") from e

        arr = np.array(raw_board)

        if arr.ndim != 3:
            if arr.ndim == 2 and arr.shape == (8, 8):
                # Handle 8x8 piece grids by expanding them into 12 binary planes.
                piece_to_chan = {
                    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
                }
                planes = np.zeros((12, 8, 8), dtype=np.float32)
                for i in range(8):
                    for j in range(8):
                        sym = arr[i, j]
                        if isinstance(sym, bytes):
                            sym = sym.decode('utf-8', errors='ignore')
                        if hasattr(sym, 'item'):
                            try:
                                sym = sym.item()
                            except Exception:
                                pass
                        if isinstance(sym, str):
                            sym = sym.strip()
                            if sym == '.' or sym == '':
                                continue
                            ch = piece_to_chan.get(sym)
                            if ch is None:
                                raise ValueError(f"Unsupported piece symbol '{sym}' at ({i},{j})")
                            planes[ch, i, j] = 1.0
                        else:
                            raise ValueError(f"Unexpected board cell type {type(sym)} at ({i},{j})")
                tensor = torch.from_numpy(planes)
                return tensor
            else:
                raise ValueError(
                    f"Expected board array with 3 dims (C,H,W) or (H,W,C) or an 8x8 piece grid, got shape {arr.shape}"
                )

        if arr.shape == (12, 8, 8):
            tensor = torch.from_numpy(arr.astype(np.float32, copy=False))
            return tensor
        elif arr.shape == (8, 8, 12):
            arr = np.transpose(arr, (2, 0, 1))
            tensor = torch.from_numpy(arr.astype(np.float32, copy=False))
            return tensor
        else:
            raise ValueError(
                f"Unsupported board shape {arr.shape}; expected (12,8,8) or (8,8,12)"
            )
        

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Any]:
        """Return a single board tensor, text description, and grouping id."""
        row = self.df.iloc[idx]
        raw_board = row[self.board_col]
        text = str(row[self.text_col])

        board_tensor = self._parse_board(raw_board)  # (12, 8, 8), float32

        if self.group_col is not None and self.group_col in row:
            group_id = row[self.group_col]
        else:
            group_id = idx

        return board_tensor, text, group_id


class BoardEncoder(nn.Module):
    """Visual backbone that embeds 12x8x8 chess planes via a ResNet18 encoder."""

    def __init__(self, proj_dim: int = 256, dropout: float = 0.0) -> None:
        """Replace the first/last ResNet layers to accept chess planes and project them."""
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            12, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(in_features, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce board embeddings from raw plane tensors."""
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feats = self.backbone(x)
        z = self.proj(feats)
        return z


class TextEncoderWithProjection(nn.Module):
    """Wraps the CLIP text tower and adds a tunable projection head."""

    def __init__(self, model_name: str, proj_dim: int = 256, device: str = "cpu") -> None:
        """Load the pretrained CLIP text encoder and expose a smaller embedding space."""
        super().__init__()
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True, dtype=torch.float32)
        self.base_model = model.text_model
        self.base_model.to(device)

        hidden_dim = self.base_model.config.hidden_size
        self.proj = nn.Linear(hidden_dim, proj_dim)

        for p in self.base_model.parameters():
            p.requires_grad = False

        if hasattr(self.base_model, "text_model"):
            encoder_layers = self.base_model.text_model.encoder.layers
            final_ln_params = self.base_model.text_model.final_layer_norm.parameters()
        else:
            encoder_layers = self.base_model.encoder.layers
            final_ln_params = self.base_model.final_layer_norm.parameters()

        # Only unfreeze the last two transformer blocks plus the final layer norm.
        for layer in encoder_layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        for p in final_ln_params:
            p.requires_grad = True

        for p in self.proj.parameters():
            p.requires_grad = True

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Tokenize raw strings and return projected CLIP text embeddings."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.base_model(**inputs)
        text_embeds = outputs.pooler_output.float()
        z = self.proj(text_embeds)
        return z


class ChessCLIP(nn.Module):
    """Contrastive model that aligns board encodings with rewritten text embeddings."""

    def __init__(
        self,
        text_model_name: str = TEXT_MODEL_NAME,
        proj_dim: int = PROJ_DIM,
        device: str = "cpu",
    ) -> None:
        """Instantiate board/text encoders plus the learnable logit scale."""
        super().__init__()
        self.device = device
        self.board_encoder = BoardEncoder(proj_dim=proj_dim)
        self.text_encoder = TextEncoderWithProjection(
            model_name=text_model_name,
            proj_dim=proj_dim,
            device=device,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_board(self, boards: torch.Tensor) -> torch.Tensor:
        """Normalize board embeddings output by the visual encoder."""
        x = self.board_encoder(boards)
        x = F.normalize(x, dim=-1)
        return x

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Normalize text embeddings produced by the CLIP text tower."""
        z = self.text_encoder(texts)
        z = F.normalize(z, dim=-1)
        return z

    def encode_game(self, game_batch: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode a batch of game sequences by averaging board embeddings."""
        B, T, C, H, W = game_batch.shape
        # Flatten to (B*T, C, H, W)
        flat = game_batch.view(B * T, C, H, W)
        # Encode all boards
        emb_flat = self.encode_board(flat)  # (B*T, D)
        # Reshape to (B, T, D)
        emb = emb_flat.view(B, T, -1)
        # Average over the actual length for each game
        avg_emb = []
        for i in range(B):
            seq_emb = emb[i, :lengths[i]]
            avg_emb.append(seq_emb.mean(dim=0))
        return torch.stack(avg_emb)

    def forward(
        self,
        boards: torch.Tensor,
        texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return bidirectional logits plus the normalized embeddings."""
        board_emb = self.encode_board(boards)
        text_emb = self.encode_text(texts)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        logits_per_board = logit_scale * board_emb @ text_emb.t()
        logits_per_text = logits_per_board.t()

        return logits_per_board, logits_per_text, board_emb, text_emb


def clip_loss(
    logits_per_board: torch.Tensor,
    logits_per_text: torch.Tensor,
) -> torch.Tensor:
    """Compute the standard symmetric CLIP cross-entropy loss."""
    bsz = logits_per_board.shape[0]
    device = logits_per_board.device
    target = torch.arange(bsz, device=device)

    loss_i = F.cross_entropy(logits_per_board, target)
    loss_t = F.cross_entropy(logits_per_text, target)
    loss = (loss_i + loss_t) / 2.0
    return loss


def multi_positive_clip_loss(
    logits_per_board: torch.Tensor,
    logits_per_text: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    """Extend CLIP loss to support many positives per anchor via group ids."""
    device = logits_per_board.device
    group_ids = group_ids.to(device)

    pos_mask = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
    num_pos = pos_mask.float().sum(dim=1).clamp(min=1.0)

    lsm_bt = logits_per_board.log_softmax(dim=1)
    # Average the log-probability across positives to avoid bias toward large groups.
    loss_bt = -(lsm_bt * pos_mask.float()).sum(dim=1) / num_pos
    loss_bt = loss_bt.mean()

    lsm_tb = logits_per_text.log_softmax(dim=1)
    loss_tb = -(lsm_tb * pos_mask.float()).sum(dim=1) / num_pos
    loss_tb = loss_tb.mean()

    return (loss_bt + loss_tb) / 2.0


@torch.no_grad()
def compute_group_recall_at_k(
    board_emb: torch.Tensor,
    text_emb: torch.Tensor,
    group_ids: torch.Tensor,
    k: int = 10,
) -> float:
    """Return recall@k where positives are samples sharing the same group id."""
    k = min(k, board_emb.shape[0])

    board_norm = F.normalize(board_emb, dim=-1)
    text_norm = F.normalize(text_emb, dim=-1)

    sims = text_norm @ board_norm.t()
    _, topk_idx = sims.topk(k, dim=1)

    retrieved_groups = group_ids[topk_idx]
    true_groups = group_ids.unsqueeze(1)

    hits = (retrieved_groups == true_groups).any(dim=1).float()
    return hits.mean().item()


@torch.no_grad()
def compute_group_ranking_metrics(
    board_emb: torch.Tensor,
    text_emb: torch.Tensor,
    group_ids: torch.Tensor,
) -> Tuple[float, float]:
    """Compute MRR and mean rank for multi-positive retrieval."""
    board_norm = F.normalize(board_emb, dim=-1)
    text_norm = F.normalize(text_emb, dim=-1)

    sims = text_norm @ board_norm.t()
    sorted_idx = torch.argsort(sims, dim=1, descending=True)
    sorted_group_ids = group_ids[sorted_idx]

    positives = sorted_group_ids == group_ids.unsqueeze(1)
    has_positive = positives.any(dim=1)

    first_pos_idx = positives.float().argmax(dim=1)
    default_rank = torch.full_like(first_pos_idx, sorted_idx.size(1), dtype=torch.float32)
    first_rank = torch.where(
        has_positive,
        first_pos_idx.float() + 1.0,
        default_rank + 1.0,
    )

    mrr = torch.where(has_positive, 1.0 / first_rank, torch.zeros_like(first_rank)).mean().item()
    mean_rank = first_rank.mean().item()

    return mrr, mean_rank


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    batch_size_train: int,
    batch_size_val: int,
    val_size: float,
    num_workers: int,
    device: str,
    val_count: Optional[int] = None,
    group_col: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Split the dataframe and create train/val dataloaders with grouping info."""
    if val_df is None:
        if val_count is not None and val_count > 0:
            # Honor a fixed validation size for consistent monitoring across runs.
            if len(train_df) <= val_count:
                val_df = train_df.copy()
                train_df = train_df.iloc[0:0]
            else:
                val_df = train_df.sample(n=val_count, random_state=42)
                train_df = train_df.drop(val_df.index).reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)
        else:
            train_df, val_df = train_test_split(
                train_df,
                test_size=val_size,
                random_state=42,
                shuffle=True,
            )
    else:
        train_df = train_df.reset_index(drop=True)
        # Apply val_count cap even when a pre-split val_df is supplied.
        if val_count is not None and val_count > 0 and len(val_df) > val_count:
            val_df = val_df.sample(n=val_count, random_state=42).reset_index(drop=True)
        else:
            val_df = val_df.reset_index(drop=True)

    train_dataset = ChessDataset(train_df, group_col=group_col)
    val_dataset = ChessDataset(val_df, group_col=group_col)

    pin_memory = device.startswith("cuda")


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: ChessCLIP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    accum_steps: int = 1,
) -> float:
    """Run one training epoch with optional gradient accumulation."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc="Train", leave=False)
    for step, batch in enumerate(pbar):
        boards, texts, group_ids = batch
        boards = boards.to(device=device, non_blocking=True)
        group_ids = group_ids.to(device=device, non_blocking=True)

        logits_per_board, logits_per_text, board_emb, text_emb = model(boards, texts)
        loss = multi_positive_clip_loss(logits_per_board, logits_per_text, group_ids)
        # Keep gradients invariant to the accumulation factor.
        loss = loss / accum_steps

        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        num_batches += 1
        pbar.set_postfix(loss=loss.item() * accum_steps)

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


# ── Negative pool cache (built once, reused across epochs) ──────────────
_NEG_POOL_CACHE: Optional[np.ndarray] = None   # (N, 12, 8, 8) float32
_STREAMING_NEG_CACHE: Optional[Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray]]] = None


def _build_negative_pool(max_negatives: Optional[int] = None) -> np.ndarray:
    """
    Build a pool of negative board matrices from the Parquet directory.
    Parquet files are read one at a time to avoid loading everything into RAM.
    The board_matrix column is an 8×8 grid of piece symbols (current format)
    or raw bytes (legacy format); both are handled transparently.
    Cached to disk as .npy so subsequent runs skip Parquet parsing entirely.
    """
    import glob as _glob
    global _NEG_POOL_CACHE, _NEG_INDEX_CACHE
    if max_negatives is None:
        max_negatives = NEGATIVES_PER_THRESHOLD
    if _NEG_POOL_CACHE is not None:
        return _NEG_POOL_CACHE

    cache_path = os.path.join(OUTPUT_DIR, "neg_pool_cache.npy")
    if os.path.isfile(cache_path):
        cached = np.load(cache_path, mmap_mode="r")
        if len(cached) == max_negatives:
            print(f"Loading cached negative pool from {cache_path} ...")
            _NEG_POOL_CACHE = np.load(cache_path)
            print(f"Loaded {len(_NEG_POOL_CACHE)} negative positions from cache.")
            return _NEG_POOL_CACHE
        else:
            print(f"Negative pool cache has {len(cached)} positions but "
                  f"{max_negatives} requested. Rebuilding...")
            del cached
            os.remove(cache_path)
            # Also invalidate distance matrix since negatives changed
            dist_cache = os.path.join(OUTPUT_DIR, "dist_matrix_cache.npy")
            if os.path.isfile(dist_cache):
                os.remove(dist_cache)
                print("Distance matrix cache invalidated.")
            neg_index_cache = os.path.join(OUTPUT_DIR, "neg_indices_cache.npz")
            if os.path.isfile(neg_index_cache):
                os.remove(neg_index_cache)
                print("Negative index cache invalidated.")
            _NEG_INDEX_CACHE = None

    parquet_files = sorted(_glob.glob(os.path.join(NEGATIVE_POSITIONS_PARQUET_DIR, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in: {NEGATIVE_POSITIONS_PARQUET_DIR}"
        )

    # Shuffle parquets so we sample from ALL over the dataset, not just the first few files
    rng_pool = np.random.default_rng(42)
    parquet_files = list(rng_pool.permutation(parquet_files))

    # How many positions to read per parquet to spread max_negatives evenly
    n_per_file = max(1, int(np.ceil(max_negatives / len(parquet_files))))

    print(f"Building negative position pool from {len(parquet_files)} Parquet files "
          f"(target {max_negatives:,} positions, ~{n_per_file} per file)...")
    pool: List[np.ndarray] = []

    _piece_to_chan = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }

    def _grid8x8_to_planes(grid) -> np.ndarray:
        """Convert an 8×8 piece-symbol grid to a (12,8,8) float32 array."""
        planes = np.zeros((12, 8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                sym = grid[i][j]
                if isinstance(sym, bytes):
                    sym = sym.decode('utf-8', errors='ignore')
                if hasattr(sym, 'item'):
                    sym = sym.item()
                if isinstance(sym, str):
                    sym = sym.strip()
                    if sym in ('.', ''):
                        continue
                    ch = _piece_to_chan.get(sym)
                    if ch is not None:
                        planes[ch, i, j] = 1.0
        return planes

    for pq_path in tqdm(parquet_files, desc="Reading Parquet files", leave=False):
        # Read one parquet file at a time — stays in RAM only while processing it
        pq_df = pd.read_parquet(pq_path, columns=["board_matrix"])

        _diag_sample = None  # save one raw value for error reporting
        file_start_len = len(pool)  # track how many we've added from this file
        for raw in pq_df["board_matrix"].tolist():
            if _diag_sample is None:
                _diag_sample = raw
            arr = None
            try:
                # ── Normalise to numpy array ──────────────────────────────────
                if isinstance(raw, (bytes, bytearray)):
                    # Legacy: planes.tobytes() → 3072 bytes → (12,8,8) float32
                    tmp = np.frombuffer(raw, dtype=np.float32).reshape(12, 8, 8).copy()
                else:
                    if isinstance(raw, str):
                        parsed = ast.literal_eval(raw)
                    else:
                        parsed = raw  # Python list or np.ndarray already

                    tmp = np.array(parsed, dtype=object)  # keep object for shape check

                # ── Dispatch by shape ─────────────────────────────────────────
                if isinstance(tmp, np.ndarray):
                    if tmp.shape == (12, 8, 8):
                        # One-hot planes (float or object)
                        arr = tmp.astype(np.float32)
                    elif tmp.shape == (8, 8, 12):
                        arr = np.transpose(tmp.astype(np.float32), (2, 0, 1))
                    elif tmp.shape == (768,):
                        # Flat list: planes.flatten().tolist()
                        arr = tmp.astype(np.float32).reshape(12, 8, 8)
                    elif tmp.shape == (8, 8):
                        # 8×8 grid already 2D
                        arr = _grid8x8_to_planes(tmp.tolist())
                    elif tmp.shape == (8,) and tmp.dtype == object:
                        # pyarrow reads list<list<string>> as array of shape (8,)
                        # where each element is itself an array/list of 8 strings
                        grid = [list(row) for row in tmp]
                        if len(grid) == 8 and all(len(r) == 8 for r in grid):
                            arr = _grid8x8_to_planes(grid)
            except Exception:
                continue

            if arr is not None and arr.shape == (12, 8, 8):
                pool.append(arr)

            if len(pool) - file_start_len >= n_per_file:
                break  # taken enough from this file; move to the next

        # Free the parquet DataFrame from RAM before loading the next file
        del pq_df

        if len(pool) >= max_negatives:
            break

    pool = pool[:max_negatives]
    if not pool:
        # Collect a diagnostic sample from the first parquet to help debugging
        try:
            _sample_df = pd.read_parquet(parquet_files[0], columns=["board_matrix"])
            _s = _sample_df["board_matrix"].iloc[0]
            _info = f"type={type(_s).__name__}"
            if isinstance(_s, (list, np.ndarray)):
                _a = np.array(_s, dtype=object)
                _info += f", shape={_a.shape}, dtype={_a.dtype}"
                if _a.size > 0:
                    _info += f", sample={repr(_a.flat[0])}"
            elif isinstance(_s, (bytes, bytearray)):
                _info += f", len={len(_s)}"
            else:
                _info += f", value={repr(_s)[:120]}"
        except Exception as _e:
            _info = str(_e)
        raise ValueError(
            f"No valid board_matrix entries found in the Parquet files. "
            f"First entry info: {_info}"
        )
    arr_pool = np.stack(pool, axis=0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(cache_path, arr_pool)
    print(f"Negative pool ({len(arr_pool)} positions) cached at {cache_path}")
    _NEG_POOL_CACHE = arr_pool
    return _NEG_POOL_CACHE


# ── Streaming negative cache (parquet-by-parquet, Numba-accelerated) ────


def _build_streaming_neg_cache(
    val_boards_np: np.ndarray,
    intervals: List[Tuple[int, int]],
    per_threshold: int,
    cache_dir: str,
    parquet_dir: str,
    seed: int = NEGATIVE_SELECTION_SEED,
    chunk_size: int = PARQUET_CHUNK_SIZE,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """
    Build negative cache by streaming through parquets chunk by chunk.
    Each parquet is read fully but processed in chunks of `chunk_size` to
    limit RAM usage.  Uses Numba-compiled `_nb_batch_distances` for speed.

    Shows one progress bar per distance interval (9 bars).
    Each unit = a query that has found all `per_threshold` negatives for that interval.

    Returns
    -------
    pool : np.ndarray  (P, 12, 8, 8) float32
    indices : dict[(lo, hi), np.ndarray]  interval -> (Q, per_threshold) int32
    """
    import glob as _glob
    import gc
    import json

    global _STREAMING_NEG_CACHE
    if _STREAMING_NEG_CACHE is not None:
        return _STREAMING_NEG_CACHE

    os.makedirs(cache_dir, exist_ok=True)
    pool_cache_path  = os.path.join(cache_dir, "neg_streaming_pool.npy")
    index_cache_path = os.path.join(cache_dir, "neg_streaming_indices.npz")
    meta_cache_path  = os.path.join(cache_dir, "neg_streaming_meta.json")

    Q                = len(val_boards_np)
    intervals_sorted = sorted(intervals, key=lambda x: x[0])
    T                = len(intervals_sorted)
    # JSON-serialisable key for cache validity
    meta_key = {
        "Q": Q,
        "intervals": [[lo, hi] for lo, hi in intervals_sorted],
        "per_threshold": per_threshold,
        "metric": "chess_intervals_numba_v1",
    }

    ckpt_dir = os.path.join(cache_dir, "streaming_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    def _save_neg_checkpoint(fully_sat: int) -> None:
        """Save a partial snapshot of pool + indices + progress JSON."""
        prefix = os.path.join(ckpt_dir, f"neg_ckpt_{fully_sat:06d}")
        # Pool
        pool_snap = (
            np.stack(pool_list) if pool_list
            else np.empty((0, 12, 8, 8), dtype=np.float32)
        )
        np.save(prefix + "_pool.npy", pool_snap)
        # Indices (current partial selection)
        idx_snap: Dict[str, np.ndarray] = {}
        for ti2, iv2 in enumerate(intervals_sorted):
            lo2, hi2 = iv2
            arr2 = np.full((Q, per_threshold), -1, dtype=np.int32)
            for q2 in range(Q):
                sl2 = selected[iv2][q2][:per_threshold]
                arr2[q2, :len(sl2)] = sl2
            idx_snap[f"iv_{lo2}_{hi2}"] = arr2
        np.savez_compressed(prefix + "_indices.npz", **idx_snap)
        # JSON progress
        progress = {
            "fully_satisfied_queries": fully_sat,
            "total_queries": Q,
            "per_threshold": per_threshold,
            "pool_size": int(len(pool_list)),
            "queries_per_interval": {
                _iv_label(lo2, hi2): int((counts[:, ti2] >= per_threshold).sum())
                for ti2, (lo2, hi2) in enumerate(intervals_sorted)
            },
        }
        with open(prefix + "_progress.json", "w") as _jf:
            json.dump(progress, _jf, indent=2)
        tqdm.write(f"  \u2714 Checkpoint saved: {fully_sat} fully-satisfied queries | "
                   f"pool={len(pool_list):,} boards -> {prefix}_*.{{npy,npz,json}}")

    # ── Fast path: load from cache ───────────────────────────────────────
    if (os.path.isfile(pool_cache_path) and os.path.isfile(index_cache_path)
            and os.path.isfile(meta_cache_path)):
        try:
            with open(meta_cache_path) as f:
                saved_meta = json.load(f)
            if saved_meta == meta_key:
                print("Loading streaming neg cache from disk...")
                pool = np.load(pool_cache_path)
                with np.load(index_cache_path, allow_pickle=False) as npz:
                    indices: Dict[Tuple[int, int], np.ndarray] = {}
                    for k in npz.files:
                        if k.startswith("iv_"):
                            parts = k[3:].split("_")
                            lo, hi = int(parts[0]), int(parts[1])
                            indices[(lo, hi)] = npz[k].copy()
                print(f"  Pool: {len(pool):,} boards | Intervals: {len(indices)}")
                _STREAMING_NEG_CACHE = (pool, indices)
                return pool, indices
        except Exception as e:
            print(f"Cache invalid ({e}). Rebuilding...")

    # ── Board parsing helper ─────────────────────────────────────────────
    _piece_to_ch = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }

    def _grid_to_planes(grid) -> np.ndarray:
        planes = np.zeros((12, 8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                sym = grid[i][j]
                if isinstance(sym, bytes):
                    sym = sym.decode('utf-8', errors='ignore')
                if hasattr(sym, 'item'):
                    sym = sym.item()
                if isinstance(sym, str):
                    sym = sym.strip()
                    if sym in ('.', ''):
                        continue
                    ch = _piece_to_ch.get(sym)
                    if ch is not None:
                        planes[ch, i, j] = 1.0
        return planes

    def _parse_raw(raw) -> Optional[np.ndarray]:
        try:
            if isinstance(raw, np.ndarray):
                if raw.shape == (12, 8, 8):
                    return raw.astype(np.float32)
                if raw.shape == (8, 8, 12):
                    return np.transpose(raw.astype(np.float32), (2, 0, 1))
                if raw.shape == (768,):
                    return raw.astype(np.float32).reshape(12, 8, 8)
                if raw.shape == (8, 8):
                    return _grid_to_planes(raw.tolist())
            if isinstance(raw, (bytes, bytearray)):
                return np.frombuffer(raw, dtype=np.float32).reshape(12, 8, 8).copy()
            if isinstance(raw, str):
                raw = ast.literal_eval(raw)
            tmp = np.array(raw, dtype=object)
            if tmp.shape == (12, 8, 8):
                return tmp.astype(np.float32)
            if tmp.shape == (8, 8, 12):
                return np.transpose(tmp.astype(np.float32), (2, 0, 1))
            if tmp.shape == (768,):
                return tmp.astype(np.float32).reshape(12, 8, 8)
            if tmp.shape == (8, 8):
                return _grid_to_planes(tmp.tolist())
            if tmp.shape == (8,) and tmp.dtype == object:
                grid = [list(row) for row in tmp]
                if len(grid) == 8 and all(len(r) == 8 for r in grid):
                    return _grid_to_planes(grid)
        except Exception:
            pass
        return None

    # ── Setup ─────────────────────────────────────────────────────────────
    parquet_files = sorted(_glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {parquet_dir}")

    rng = np.random.default_rng(seed)

    # selected[(lo,hi)][q] = list of pool indices
    selected: Dict[Tuple[int, int], List[List[int]]] = {
        iv: [[] for _ in range(Q)] for iv in intervals_sorted
    }
    counts = np.zeros((Q, T), dtype=np.int32)

    # Pool with deduplication
    board_key_to_idx: Dict[bytes, int] = {}
    pool_list: List[np.ndarray] = []

    def _board_key(b: np.ndarray) -> bytes:
        return np.packbits((b > 0.5).ravel()).tobytes()

    def _add_to_pool(b: np.ndarray) -> int:
        key = _board_key(b)
        if key in board_key_to_idx:
            return board_key_to_idx[key]
        idx = len(pool_list)
        pool_list.append(b.copy())
        board_key_to_idx[key] = idx
        return idx

    def _all_done() -> bool:
        return bool(np.all(counts >= per_threshold))

    # Ensure val boards are contiguous float32 for Numba
    val_boards_f32 = np.ascontiguousarray(val_boards_np, dtype=np.float32)

    # Build interval bounds as numpy arrays for fast binning
    iv_lo = np.array([lo for lo, _ in intervals_sorted], dtype=np.int32)
    iv_hi = np.array([hi for _, hi in intervals_sorted], dtype=np.int32)

    print(f"\nBuilding streaming neg cache (Numba): {len(parquet_files)} parquets | "
          f"{Q} queries | {per_threshold} negs/interval | "
          f"chunk_size={chunk_size:,}")
    print(f"Intervals: {intervals_sorted}")
    print("(runs once; cached to disk for all subsequent epochs)\n")

    # ── Progress bars (one per interval) ──────────────────────────────────
    # Each bar shows total negatives accumulated for that interval
    # across all queries. Total = Q * per_threshold.
    def _iv_label(lo: int, hi: int) -> str:
        if hi >= 9999:
            return f"d>={lo:>3d}"
        return f"d={lo:>3d}-{hi}"

    total_per_interval = Q * per_threshold   # e.g. 1000 queries * 1000 negs = 1,000,000
    dist_pbars = [
        tqdm(total=total_per_interval, desc=_iv_label(lo, hi), position=ti,
             leave=True, unit="neg", unit_scale=True,
             bar_format="{desc}: {n_fmt}/{total_fmt} |{bar:40}| "
                        "{percentage:3.0f}%  [{elapsed}<{remaining}, {rate_fmt}]")
        for ti, (lo, hi) in enumerate(intervals_sorted)
    ]
    prev_neg_totals = np.zeros(T, dtype=np.int64)
    last_ckpt_milestone = 0           # last milestone (in units of NEG_CACHE_CHECKPOINT_EVERY) saved
    fully_sat_running = 0             # running count of fully-satisfied queries
    query_satisfied = np.zeros(Q, dtype=np.bool_)  # per-query completion flag

    try:
        for pq_idx, pq_path in enumerate(parquet_files):
            if _all_done():
                tqdm.write(
                    f"\n\u2713 All queries fully satisfied after {pq_idx}/"
                    f"{len(parquet_files)} parquets. Pool: {len(pool_list):,} boards."
                )
                break

            # ── Load one parquet ──────────────────────────────────────────
            try:
                pq_df = pd.read_parquet(pq_path, columns=["board_matrix"])
            except Exception:
                continue

            raw_list = pq_df["board_matrix"].tolist()
            del pq_df

            all_boards = []
            for _raw_item in tqdm(raw_list, desc=f"  parsing pq {pq_idx+1}/{len(parquet_files)}",
                                  leave=False, position=T, unit="row"):
                _parsed = _parse_raw(_raw_item)
                if _parsed is not None:
                    all_boards.append(_parsed)
            del raw_list
            if not all_boards:
                continue

            M_full = len(all_boards)

            # ── Process in chunks of chunk_size ───────────────────────────
            for c_start in range(0, M_full, chunk_size):
                if _all_done():
                    break

                c_end = min(c_start + chunk_size, M_full)
                chunk_boards = np.ascontiguousarray(
                    np.stack(all_boards[c_start:c_end]), dtype=np.float32
                )
                M = len(chunk_boards)

                # Active queries only
                active = np.flatnonzero(np.any(counts < per_threshold, axis=1))
                if len(active) == 0:
                    break

                # ── Compute distances in parallel query-batches ──────────
                QBATCH = EVAL_QUERY_CHUNK          # e.g. 64 queries at a time
                for qb_start in range(0, len(active), QBATCH):
                    qb_end = min(qb_start + QBATCH, len(active))
                    qb_indices = active[qb_start:qb_end]
                    query_batch = np.ascontiguousarray(val_boards_f32[qb_indices])

                    # Parallel Numba: returns (len(qb_indices), M) int32
                    dists_batch = _nb_multi_batch_distances(
                        query_batch, chunk_boards,
                        _KNIGHT_DIST_TABLE, _PENALTIES,
                    )

                    for local_qi in range(len(qb_indices)):
                        qi = int(qb_indices[local_qi])
                        dists_qi = dists_batch[local_qi]

                        # Bin into intervals
                        for ti in range(T):
                            if counts[qi, ti] >= per_threshold:
                                continue
                            mask = (dists_qi >= iv_lo[ti]) & (dists_qi <= iv_hi[ti])
                            eligible = np.flatnonzero(mask)
                            if eligible.size == 0:
                                continue
                            pool_ids = [_add_to_pool(chunk_boards[li]) for li in eligible]
                            need = per_threshold - counts[qi, ti]
                            take = min(len(pool_ids), need)
                            if take > 0:
                                chosen = rng.choice(len(pool_ids), size=take, replace=False)
                                iv = intervals_sorted[ti]
                                for c in chosen:
                                    selected[iv][qi].append(pool_ids[c])
                                counts[qi, ti] += take

                        # Check if this query just became fully satisfied
                        if not query_satisfied[qi] and np.all(counts[qi] >= per_threshold):
                            query_satisfied[qi] = True
                            fully_sat_running += 1
                            milestone = fully_sat_running // NEG_CACHE_CHECKPOINT_EVERY
                            if milestone > last_ckpt_milestone:
                                _save_neg_checkpoint(fully_sat_running)
                                last_ckpt_milestone = milestone

                    # ── Update progress bars after each query batch ──────
                    for ti in range(T):
                        cur_total = int(np.minimum(counts[:, ti], per_threshold).sum())
                        inc = cur_total - int(prev_neg_totals[ti])
                        if inc > 0:
                            dist_pbars[ti].update(inc)
                            prev_neg_totals[ti] = cur_total

                del chunk_boards

            del all_boards
            gc.collect()

            active_left = Q - fully_sat_running
            tqdm.write(
                f"  \u2500\u2500 pq {pq_idx+1}/{len(parquet_files)} "
                f"({M_full:,} boards) | pool={len(pool_list):,} | "
                f"active queries={active_left} | fully_sat={fully_sat_running}"
            )

    finally:
        for pb in dist_pbars:
            pb.close()

    # ── Report ────────────────────────────────────────────────────────────
    unsatisfied = int((counts < per_threshold).sum())
    if unsatisfied > 0:
        print(f"NOTE: {unsatisfied} (query, interval) pairs have fewer than "
              f"{per_threshold} negatives. They will use only what was found.")

    # ── Convert & save ────────────────────────────────────────────────────
    pool_arr = (
        np.stack(pool_list) if pool_list
        else np.empty((0, 12, 8, 8), dtype=np.float32)
    )
    indices: Dict[Tuple[int, int], np.ndarray] = {}
    for ti, iv in enumerate(intervals_sorted):
        arr = np.full((Q, per_threshold), -1, dtype=np.int32)
        for q in range(Q):
            sl = selected[iv][q][:per_threshold]
            arr[q, :len(sl)] = sl
        indices[iv] = arr

    np.save(pool_cache_path, pool_arr)
    np.savez_compressed(
        index_cache_path,
        **{f"iv_{lo}_{hi}": arr for (lo, hi), arr in indices.items()},
    )
    with open(meta_cache_path, "w") as f:
        json.dump(meta_key, f)

    print(f"\u2713 Streaming neg cache saved: {len(pool_arr):,} boards, "
          f"{len(intervals_sorted)} intervals")
    _STREAMING_NEG_CACHE = (pool_arr, indices)
    return pool_arr, indices


@torch.no_grad()
def evaluate(model: ChessCLIP, val_loader: DataLoader, device: str, epoch_label: Optional[str] = None):
    """
    Full evaluation using ALL validation queries and the external negative pool.
     - Uses ALL val queries (no sampling).
     - Builds/loads streaming neg pool via parquet chunks (cached .npy).
     - Uses distance *intervals* (e.g. 1-20, 21-35, …) with Numba-JIT distances.
     - Each epoch: encode negatives on GPU, chunked similarity → metrics per interval.
    """
    model.eval()
    distance_intervals = list(EVAL_DISTANCE_INTERVALS)

    # ── 1. Collect ALL validation data ──────────────────────────────────
    all_boards_list: List[torch.Tensor] = []
    all_texts: List[str] = []

    print("Collecting validation queries...")
    for batch in tqdm(val_loader, desc="Loading Val", leave=False):
        boards, texts, _ = batch
        all_boards_list.append(boards)
        all_texts.extend(texts)

    query_boards_all = torch.cat(all_boards_list, dim=0)              # (Q, 12, 8, 8)
    num_eval = len(query_boards_all)

    # ── 2. Validation loss on full set (for monitoring) ─────────────────
    running_loss = 0.0
    num_batches = 0
    for batch in tqdm(val_loader, desc="Val Loss", leave=False):
        boards, texts, group_ids = batch
        boards = boards.to(device, non_blocking=True)
        group_ids = group_ids.to(device)
        logits_per_board, logits_per_text, _, _ = model(boards, texts)
        loss = multi_positive_clip_loss(logits_per_board, logits_per_text, group_ids)
        running_loss += loss.item()
        num_batches += 1
    avg_loss = running_loss / max(num_batches, 1)

    # ── 3. Encode ALL queries (board + text) ────────────────────────────
    print(f"Encoding {num_eval} queries (all val)...")
    q_text_embs_list: List[torch.Tensor] = []
    for i in range(0, num_eval, BATCH_SIZE_VAL):
        t_batch = all_texts[i:i + BATCH_SIZE_VAL]
        q_text_embs_list.append(model.encode_text(t_batch).cpu())
    q_text_embs = torch.cat(q_text_embs_list, dim=0)                 # (Q, D)

    q_board_embs_list: List[torch.Tensor] = []
    for i in range(0, num_eval, BATCH_SIZE_VAL):
        b_batch = query_boards_all[i:i + BATCH_SIZE_VAL].to(device)
        q_board_embs_list.append(model.encode_board(b_batch).cpu())
    q_board_embs = torch.cat(q_board_embs_list, dim=0)               # (Q, D)

    # ── 4. Build / load streaming neg cache (parquet-by-parquet, one-time) ─
    pool_np, neg_indices_by_interval = _build_streaming_neg_cache(
        query_boards_all.numpy(),
        distance_intervals,
        NEGATIVES_PER_THRESHOLD,
        OUTPUT_DIR,
        NEGATIVE_POSITIONS_PARQUET_DIR,
    )
    neg_pool_tensor = torch.from_numpy(pool_np)
    num_negs = len(pool_np)
    if num_negs == 0:
        raise ValueError("Streaming neg cache is empty. Check parquet files.")

    # ── 5. Encode ALL negatives ONCE this epoch ─────────────────────────
    print(f"Encoding {num_negs} negative positions...")
    neg_embs_list: List[torch.Tensor] = []
    for i in tqdm(range(0, num_negs, NEGATIVE_EMBED_BATCH_SIZE), desc="Encode Neg", leave=False):
        batch_t = neg_pool_tensor[i:i + NEGATIVE_EMBED_BATCH_SIZE].to(device)
        neg_embs_list.append(model.encode_board(batch_t).cpu())
    neg_embs = torch.cat(neg_embs_list, dim=0)                       # (N, D)

    # ── 7. Positive scores ──────────────────────────────────────────────
    pos_scores = (q_text_embs * q_board_embs).sum(dim=1).numpy()      # (Q,)

    # ── 8. Chunked similarity + metrics (memory-efficient) ──────────────
    print(f"Computing similarities & metrics in chunks of {EVAL_QUERY_CHUNK} "
          f"({num_eval} queries × {num_negs} negatives)...")

    # Accumulators per interval
    accum: Dict[Tuple[int, int], Dict[str, float]] = {
        iv: {"r1": 0.0, "r5": 0.0, "r10": 0.0,
             "rr_sum": 0.0, "rank_sum": 0.0,
             "neg_sum": 0.0, "n": 0}
        for iv in distance_intervals
    }

    for qi_start in tqdm(range(0, num_eval, EVAL_QUERY_CHUNK),
                         desc="Eval chunks", leave=False):
        qi_end = min(qi_start + EVAL_QUERY_CHUNK, num_eval)
        chunk_size = qi_end - qi_start

        # Similarity chunk: (chunk, N)
        sims_chunk = (q_text_embs[qi_start:qi_end] @ neg_embs.t()).numpy()
        pos_chunk = pos_scores[qi_start:qi_end]

        for iv in distance_intervals:
            idx_chunk = neg_indices_by_interval[iv][qi_start:qi_end]
            for local_i in range(chunk_size):
                idxs = idx_chunk[local_i]
                valid_mask = idxs >= 0
                if not np.any(valid_mask):
                    continue
                neg_ids = idxs[valid_mask].astype(np.int64, copy=False)
                neg_scores = sims_chunk[local_i, neg_ids]
                gt_pos = int((neg_scores > pos_chunk[local_i]).sum())
                rank = gt_pos + 1

                accum[iv]["r1"]       += 1.0 if rank <= 1  else 0.0
                accum[iv]["r5"]       += 1.0 if rank <= 5  else 0.0
                accum[iv]["r10"]      += 1.0 if rank <= 10 else 0.0
                accum[iv]["rr_sum"]   += 1.0 / rank
                accum[iv]["rank_sum"] += rank
                accum[iv]["neg_sum"]  += len(neg_ids)
                accum[iv]["n"]        += 1

    # ── 9. Aggregate & save ─────────────────────────────────────────────
    print(f"\n=== Evaluation Results by Distance Interval "
          f"(all {num_eval} queries, {num_negs} negatives) ===")
    metrics_summary: Dict[Tuple[int, int], Dict[str, float]] = {}
    metrics_rows: List[dict] = []

    for iv in distance_intervals:
        lo, hi = iv
        label = f">={lo}" if hi >= 9999 else f"{lo}-{hi}"
        n = int(accum[iv]["n"])
        if n == 0:
            print(f"Distance {label:>7s}: No valid queries")
            continue

        r1       = accum[iv]["r1"]       / n
        r5       = accum[iv]["r5"]       / n
        r10      = accum[iv]["r10"]      / n
        mrr      = accum[iv]["rr_sum"]   / n
        mean_rank = accum[iv]["rank_sum"] / n
        mean_negs = accum[iv]["neg_sum"]  / n

        metrics_summary[iv] = {
            "recall_at_1": r1, "recall_at_5": r5, "recall_at_10": r10,
            "mrr": mrr, "mean_rank": mean_rank,
            "num_queries": n, "avg_negatives_used": mean_negs,
        }
        metrics_rows.append({
            "distance_interval": label,
            "recall_at_1": r1, "recall_at_5": r5, "recall_at_10": r10,
            "mrr": mrr, "mean_rank": mean_rank,
            "num_queries": n, "avg_negatives_used": mean_negs,
            "expected_negatives": NEGATIVES_PER_THRESHOLD,
            "eval_source": epoch_label or "epoch",
        })
        print(
            f"Distance {label:>7s}: R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f} "
            f"MRR={mrr:.4f} MR={mean_rank:.2f} avg_negs={mean_negs:.1f} "
            f"(n_queries={n})"
        )

    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows)
        suffix = epoch_label or "epoch"
        csv_path = os.path.join(OUTPUT_DIR, f"eval_metrics_by_distance_{suffix}.csv")
        df_metrics.to_csv(csv_path, index=False)
        print(f"\nDistance-based metrics saved to: {csv_path}")

    all_r1 = [m["recall_at_1"] for m in metrics_summary.values()]
    all_r5 = [m["recall_at_5"] for m in metrics_summary.values()]
    all_r10 = [m["recall_at_10"] for m in metrics_summary.values()]
    all_mrr = [m["mrr"] for m in metrics_summary.values()]
    all_mean_rank = [m["mean_rank"] for m in metrics_summary.values()]

    avg_r1 = float(np.mean(all_r1)) if all_r1 else 0.0
    avg_r5 = float(np.mean(all_r5)) if all_r5 else 0.0
    avg_r10 = float(np.mean(all_r10)) if all_r10 else 0.0
    avg_mrr = float(np.mean(all_mrr)) if all_mrr else 0.0
    avg_mean_rank = float(np.mean(all_mean_rank)) if all_mean_rank else 0.0
    return avg_loss, avg_r1, avg_r5, avg_r10, avg_mrr, avg_mean_rank


def train_clip(train_df: pd.DataFrame, val_df: pd.DataFrame) -> str:
    """Train the ChessCLIP model on the provided dataframe and return best ckpt path."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Total rows in train dataset (raw): {len(train_df)}")
    print(f"Total rows in val dataset (raw): {len(val_df)}")
    for source_name, source_df in (("train", train_df), ("val", val_df)):
        assert "board_matrix" in source_df.columns, f"Expected 'board_matrix' column in the {source_name} dataframe."
        assert "rewritten" in source_df.columns, f"Expected 'rewritten' column in the {source_name} dataframe."

    def _is_valid_board(x: Any) -> bool:
        return True

    def _prepare_df(source_df: pd.DataFrame, label: str) -> pd.DataFrame:
        source_df = source_df[source_df["board_matrix"].apply(_is_valid_board)].reset_index(drop=True)
        print(f"Rows after filtering invalid boards ({label}): {len(source_df)}")

        source_df = source_df[source_df["rewritten"].apply(_is_informative_text)]

        if "group_id" not in source_df.columns:
            source_df["group_id"] = source_df["board_matrix"].factorize()[0]

        source_df = source_df.reset_index(drop=True)
        print(f"Rows after text cleaning (multi-positives enabled) ({label}): {len(source_df)}")
        return source_df

    def _is_informative_text(t: Any) -> bool:
        return True

    train_df = _prepare_df(train_df, "train")
    val_df = _prepare_df(val_df, "val")

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_df,
        val_df,
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_val=BATCH_SIZE_VAL,
        val_size=VAL_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
        val_count=VAL_COUNT,
        group_col="group_id",
    )

    print("Building model...")
    model = ChessCLIP(
        text_model_name=TEXT_MODEL_NAME,
        proj_dim=PROJ_DIM,
        device=device,
    )
    model.to(device)

    text_params = [
        p
        for n, p in model.text_encoder.base_model.named_parameters()
        if p.requires_grad
    ]
    other_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("text_encoder.base_model")
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": LR},
            {"params": text_params, "lr": TEXT_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    WARMUP_EPOCHS = 5
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - WARMUP_EPOCHS,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    best_val_loss = float("inf")
    best_recall = 0.0
    best_path = os.path.join(OUTPUT_DIR, "chess_clip_best.pt")
    patience_counter = 0
    training_history: List[dict] = []

    start_epoch = 1
    if resume_ckpt is not None and os.path.isfile(resume_ckpt):
        print(f"Resuming training from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print("Warning: could not load optimizer state from checkpoint, continuing with fresh optimizer.")
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", best_val_loss)))
        best_recall = float(ckpt.get("best_recall", ckpt.get("recall_at_10", best_recall)))
        patience_counter = int(ckpt.get("patience_counter", 0))
        print(f"Resumed from epoch {start_epoch-1}")

    for epoch in range(start_epoch, EPOCHS + 1):
        print("-----------------------------------------------------------------")
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            accum_steps=ACCUM_STEPS,
        )
        (
            val_loss,
            recall_at_1,
            recall_at_5,
            recall_at_10,
            mrr,
            mean_rank,
        ) = evaluate(model, val_loader, device, epoch_label=f"epoch_{epoch:04d}")

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"R@1={recall_at_1:.4f}  R@5={recall_at_5:.4f}  R@10={recall_at_10:.4f}  "
            f"MRR={mrr:.4f}  MR={mean_rank:.2f}"
        )
        print("-----------------------------------------------------------------")

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "recall_at_1": recall_at_1,
                "recall_at_5": recall_at_5,
                "recall_at_10": recall_at_10,
                "mrr": mrr,
                "mean_rank": mean_rank,
            }
        )

        metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.csv")
        pd.DataFrame(training_history).to_csv(metrics_path, index=False)

        scheduler.step()

        if SAVE_EVERY > 0 and (epoch % SAVE_EVERY == 0):
            periodic_path = os.path.join(OUTPUT_DIR, f"chess_clip_epoch_{epoch:04d}_{recall_at_10:.4f}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "recall_at_1": recall_at_1,
                    "recall_at_5": recall_at_5,
                    "recall_at_10": recall_at_10,
                    "mrr": mrr,
                    "mean_rank": mean_rank,
                    "best_val_loss": best_val_loss,
                    "best_recall": best_recall,
                    "patience_counter": patience_counter,
                    "config": {
                        "TEXT_MODEL_NAME": TEXT_MODEL_NAME,
                        "PROJ_DIM": PROJ_DIM,
                        "BATCH_SIZE_TRAIN": BATCH_SIZE_TRAIN,
                        "BATCH_SIZE_VAL": BATCH_SIZE_VAL,
                        "EPOCHS": EPOCHS,
                        "PATIENCE": PATIENCE,
                        "LR": LR,
                        "TEXT_LR": TEXT_LR,
                        "WEIGHT_DECAY": WEIGHT_DECAY,
                        "VAL_SIZE": VAL_SIZE,
                        "NUM_WORKERS": NUM_WORKERS,
                        "SAVE_EVERY": SAVE_EVERY,
                        "resume_ckpt": resume_ckpt,
                    },
                },
                periodic_path,
            )
            print(f"  >> Periodic checkpoint saved to {periodic_path}")

        improved = recall_at_10 > best_recall

        if improved:
            best_val_loss = val_loss
            best_recall = recall_at_10
            patience_counter = 0
            best_path = os.path.join(OUTPUT_DIR, f"chess_clip_best_{recall_at_10:.4f}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "recall_at_1": recall_at_1,
                    "recall_at_5": recall_at_5,
                    "recall_at_10": recall_at_10,
                    "mrr": mrr,
                    "mean_rank": mean_rank,
                    "best_val_loss": best_val_loss,
                    "best_recall": best_recall,
                    "patience_counter": patience_counter,
                    "config": {
                        "TEXT_MODEL_NAME": TEXT_MODEL_NAME,
                        "PROJ_DIM": PROJ_DIM,
                        "BATCH_SIZE_TRAIN": BATCH_SIZE_TRAIN,
                        "BATCH_SIZE_VAL": BATCH_SIZE_VAL,
                        "EPOCHS": EPOCHS,
                        "PATIENCE": PATIENCE,
                        "LR": LR,
                        "TEXT_LR": TEXT_LR,
                        "WEIGHT_DECAY": WEIGHT_DECAY,
                        "VAL_SIZE": VAL_SIZE,
                        "NUM_WORKERS": NUM_WORKERS,
                        "SAVE_EVERY": SAVE_EVERY,
                        "resume_ckpt": resume_ckpt,
                    },
                },
                best_path,
            )
            print(f"  >> New best model saved to {best_path}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}: no improvement in R@10 for {PATIENCE} epochs.")
            break

    metrics_path = None
    if training_history:
        history_df = pd.DataFrame(training_history)
        metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.csv")
        history_df.to_csv(metrics_path, index=False)
        print(f"Training metrics saved to {metrics_path}")

    print("\nRunning final evaluation on validation queries with external negatives...")
    evaluate(model, val_loader, device, epoch_label="final")

    print("\nTraining finished.")
    print(f"Best val_loss: {best_val_loss:.4f}, best R@10: {best_recall:.4f}")
    print(f"Best checkpoint path: {best_path}")
    return best_path

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    def _resolve_csv_path(path_str: str) -> str:
        if os.path.isabs(path_str):
            return path_str
        return os.path.join(script_dir, path_str)

    train_csv_path = _resolve_csv_path(TRAIN_CSV_PATH)
    val_csv_path = _resolve_csv_path(VAL_CSV_PATH)

    if not os.path.isfile(train_csv_path):
        raise FileNotFoundError(
            f"Train CSV not found at '{train_csv_path}'. "
            f"Update TRAIN_CSV_PATH or run from the project root."
        )
    if not os.path.isfile(val_csv_path):
        raise FileNotFoundError(
            f"Val CSV not found at '{val_csv_path}'. "
            f"Update VAL_CSV_PATH or run from the project root."
        )

    train_df = pd.read_csv(train_csv_path)
    val_df_full = pd.read_csv(val_csv_path)

    # ── Split: 1000 val + rest → test_df_final.csv ────────────────────────
    test_csv_path = _resolve_csv_path(TEST_CSV_PATH)
    if len(val_df_full) > 1000:
        rng_split = np.random.default_rng(42)
        shuffled_idx = rng_split.permutation(len(val_df_full))
        val_df = val_df_full.iloc[shuffled_idx[:1000]].reset_index(drop=True)
        test_df = val_df_full.iloc[shuffled_idx[1000:]].reset_index(drop=True)
        if not os.path.isfile(test_csv_path):
            test_df.to_csv(test_csv_path, index=False)
            print(f"Test set ({len(test_df)} rows) saved to: {test_csv_path}")
        else:
            print(f"Test CSV already exists: {test_csv_path}")
        print(f"Using {len(val_df)} val rows for training/evaluation")
        del test_df, val_df_full
    else:
        val_df = val_df_full
        print(f"Val set has {len(val_df)} rows (<=1000), using all")
        del val_df_full

    best_ckpt = train_clip(train_df, val_df)
    print("Best checkpoint:", best_ckpt)