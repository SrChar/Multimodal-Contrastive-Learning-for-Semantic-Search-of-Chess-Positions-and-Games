"""
Training script for aligning full chess games (sequences of board positions)
with textual descriptions using frozen base encoders from the position-text
CLIP model (see `train_positions.py`) and lightweight Transformer poolers.

Main characteristics:
- Transformer pooler for game boards with d_model=256, n_heads=4,
  num_layers=2, dim_feedforward=1024, dropout=0.2.
- Learned [CLS] token (256-d) prepended to board sequences.
- Optional text encoder fine-tuning for the last two transformer blocks and
  final LayerNorm.
- Symmetric CLIP/InfoNCE loss with cosine similarity and fixed temperature.
- AdamW optimization and CSV metrics/checkpoint tracking.
- All configuration through module-level variables (no CLI args).
"""

import os
import random  # for sampling one text per game
from typing import Any, List, Optional, Tuple

import ast
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import amp

# Reuse architectures from the position-text CLIP training
from train_positions import ChessCLIP, TEXT_MODEL_NAME, PROJ_DIM


# =========================================================
# Config (edit in-code, no CLI)
# =========================================================

# Path to the pretrained position-text CLIP checkpoint.
PRETRAINED_CLIP_CKPT = "checkpoints/position_model/chess_clip_best_0.2481.pt"

OUTPUT_DIR = "checkpoints/game_models"
MODEL_PREFIX = "chess_games_clip"
METRICS_CSV_NAME = "training_metrics_games.csv"

BATCH_SIZE = 256
EPOCHS = 200
PATIENCE = 8
LR = 1e-4
WEIGHT_DECAY = 0.05
VAL_SIZE = 0.1
VAL_COUNT: Optional[int] = None
NUM_WORKERS = 0
TRAIN_CSV_PATH = "data/final/train_df_final.csv"
VAL_CSV_PATH = "data/final/val_df_final.csv"

# Transformer pooler hyperparameters
POOLER_D_MODEL = 256
POOLER_N_HEADS = 4
POOLER_NUM_LAYERS = 2
POOLER_DIM_FF = 1024
POOLER_DROPOUT = 0.2
USE_LAYER_NORM = True

# Fine-tuning controls
FINETUNE_TEXT_ENCODER = False  # set True to unfreeze last text encoder blocks

# Loss temperature
TEMPERATURE = 0.07

# Dataset columns (after grouping the flat position dataset used in train.py)
BOARD_SEQ_COL = "game_boards"    # generated list/sequence of board matrices per game
TEXT_SEQ_COL = "game_texts"      # generated list/sequence of rewritten texts per game
GROUP_COL: Optional[str] = "group_id"   # game-level id after grouping

# Checkpoint saving cadence
SAVE_EVERY = 1

# Memory controls
USE_AMP = True  # mixed precision to reduce VRAM (requires CUDA)
MAX_GAME_LEN: Optional[int] = 140  # truncate sequences per game to at most this length
MIN_GAME_LEN: Optional[int] = 4    # filter out games with fewer positions than this
ENCODE_CHUNK_SIZE_BOARDS = 512    # process board positions in chunks when encoding
ENCODE_CHUNK_SIZE_TEXTS = 512     # process text comments in chunks when encoding


# =========================================================
# Dataset
# =========================================================


class ChessGamesDataset(Dataset):
	"""
	Dataset for (game_board_sequence, text_sequence) pairs.

	Expected columns in the dataframe:
	- BOARD_SEQ_COL: iterable of board positions for the game. Each position
	  can be a list/np.array shaped (12,8,8) or (8,8,12), or an 8x8 grid of
	  piece symbols ('.', 'r', 'N', etc.). Strings will be parsed with
	  ast.literal_eval.
	- TEXT_SEQ_COL: iterable of strings (one or more comments/descriptions).
	- GROUP_COL (optional): identifier for multi-positive handling. If None,
	  each row is its own group.
	"""

	def __init__(
		self,
		df: pd.DataFrame,
		board_seq_col: str = BOARD_SEQ_COL,
		text_seq_col: str = TEXT_SEQ_COL,
		group_col: Optional[str] = GROUP_COL,
	) -> None:
		"""Store references to the grouped dataframe and relevant column names."""
		self.df = df.reset_index(drop=True)
		self.board_seq_col = board_seq_col
		self.text_seq_col = text_seq_col
		self.group_col = group_col

	def __len__(self) -> int:
		"""Return the number of grouped games in the dataset."""
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
				return torch.from_numpy(planes)
			raise ValueError(f"Expected board array with 3 dims or 8x8 grid, got shape {arr.shape}")

		if arr.shape == (12, 8, 8):
			return torch.from_numpy(arr.astype(np.float32, copy=False))
		if arr.shape == (8, 8, 12):
			arr = np.transpose(arr, (2, 0, 1))
			return torch.from_numpy(arr.astype(np.float32, copy=False))
		raise ValueError(f"Unsupported board shape {arr.shape}; expected (12,8,8) or (8,8,12)")

	def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[str], Any]:
		"""Return the list of board tensors, ONE text query, and group id for one game."""
		row = self.df.iloc[idx]
		raw_board_seq = row[self.board_seq_col]
		raw_text_seq = row[self.text_seq_col]

		# Boards: keep full sequence (game-level context)
		board_list = list(raw_board_seq)
		board_tensors = [self._parse_board(b) for b in board_list]

		# Texts: pick ONE text for this game (closest to a user query)
		if isinstance(raw_text_seq, (list, tuple)):
			if len(raw_text_seq) == 0:
				raise ValueError("Empty text sequence for game")
			chosen_text = random.choice(raw_text_seq)  # or raw_text_seq[0] for deterministic
		else:
			chosen_text = raw_text_seq

		text_list = [str(chosen_text)]

		if self.group_col is not None and self.group_col in row:
			group_id = row[self.group_col]
		else:
			group_id = idx

		return board_tensors, text_list, group_id


def collate_games(batch):
	"""Keeps variable-length sequences; padding happens after encoding."""
	board_seqs, text_seqs, group_ids = zip(*batch)
	group_ids = torch.tensor(group_ids, dtype=torch.long)
	return list(board_seqs), list(text_seqs), group_ids


def build_game_dataframe_from_positions(df_raw: pd.DataFrame) -> pd.DataFrame:
	"""
	Transform the flat position dataset used in train.py into a game-level dataset.
	Expects columns: 'game_id', 'board_matrix', 'rewritten'.
	Returns a dataframe with columns: BOARD_SEQ_COL, TEXT_SEQ_COL, GROUP_COL.
	The order within each game is preserved from the original dataframe order.
	"""

	required_cols = {"board_matrix", "rewritten"}
	missing = required_cols - set(df_raw.columns)
	if missing:
		raise ValueError(f"Missing columns in input df: {missing}")

	# Keep only necessary cols and drop rows with NaNs in critical fields
	df_clean = df_raw.dropna(subset=["board_matrix", "rewritten"]).copy()

	# If no game_id column, create one per row (no grouping)
	if "game_id" not in df_clean.columns:
		df_clean["game_id"] = df_clean.index

	# Preserve original order inside each game by index ordering
	grouped = []
	for gid, grp in df_clean.groupby("game_id", sort=False):
		grp_sorted = grp.sort_index()
		boards = list(grp_sorted["board_matrix"].values)
		texts = list(grp_sorted["rewritten"].astype(str).values)
		grouped.append({
			"game_id": gid,
			BOARD_SEQ_COL: boards,
			TEXT_SEQ_COL: texts,
			GROUP_COL: gid,
		})

	# Filter out games with more positions than MAX_GAME_LEN or fewer than MIN_GAME_LEN
	if MAX_GAME_LEN is not None or MIN_GAME_LEN is not None:
		grouped = [g for g in grouped if 
			(MAX_GAME_LEN is None or len(g[BOARD_SEQ_COL]) <= MAX_GAME_LEN) and
			(MIN_GAME_LEN is None or len(g[BOARD_SEQ_COL]) >= MIN_GAME_LEN)]

	return pd.DataFrame(grouped)


# =========================================================
# Model components
# =========================================================


class TransformerPooler(nn.Module):
	"""Lightweight Transformer encoder with a learnable CLS token for pooling."""

	def __init__(
		self,
		d_model: int = POOLER_D_MODEL,
		n_heads: int = POOLER_N_HEADS,
		num_layers: int = POOLER_NUM_LAYERS,
		dim_feedforward: int = POOLER_DIM_FF,
		dropout: float = POOLER_DROPOUT,
		use_layernorm: bool = USE_LAYER_NORM,
	) -> None:
		"""Configure the Transformer encoder stack and projection hyperparameters."""
		super().__init__()
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=n_heads,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
		self.use_layernorm = use_layernorm
		self.layernorm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

	def forward(self, seq_emb: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
		"""
		seq_emb: (B, L, D)
		pad_mask: (B, L) with True for padding positions
		Returns: (B, D) normalized pooled embeddings
		"""
		bsz = seq_emb.size(0)
		cls = self.cls_token.expand(bsz, -1, -1)
		x = torch.cat([cls, seq_emb], dim=1)
		cls_pad = torch.zeros((bsz, 1), dtype=pad_mask.dtype, device=pad_mask.device)
		pad_mask = torch.cat([cls_pad, pad_mask], dim=1).bool()
		out = self.encoder(x, src_key_padding_mask=pad_mask)
		pooled = out[:, 0]
		pooled = self.layernorm(pooled)
		pooled = F.normalize(pooled, dim=-1)
		return pooled


class ChessGamesCLIP(nn.Module):
	"""
	Frozen base encoders (from position-text CLIP) + trainable Transformer poolers
	for full-game representation and textual summary representation.
	"""

	def __init__(self, base_ckpt: Optional[str], device: str) -> None:
		"""Load the frozen position-text CLIP backbone and initialize poolers."""
		super().__init__()
		self.device = device
		if base_ckpt is not None:
			self.base_clip = self._load_frozen_base(base_ckpt, device)
		else:
			self.base_clip = ChessCLIP(text_model_name=TEXT_MODEL_NAME, proj_dim=PROJ_DIM, device=device)
		self.board_pooler = TransformerPooler().to(device)

	def _load_frozen_base(self, ckpt_path: str, device: str) -> ChessCLIP:
		"""Restore the pretrained ChessCLIP weights and freeze all parameters."""
		model = ChessCLIP(text_model_name=TEXT_MODEL_NAME, proj_dim=PROJ_DIM, device=device)
		if not os.path.isfile(ckpt_path):
			raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
		# Explicitly set weights_only=False to preserve current behavior and silence warning
		state = torch.load(ckpt_path, map_location=device, weights_only=False)
		if "model_state_dict" in state:
			state = state["model_state_dict"]
		model.load_state_dict(state, strict=False)
		model.to(device)
		model.eval()

		# 1) Freeze everything
		for p in model.parameters():
			p.requires_grad = False

		if FINETUNE_TEXT_ENCODER:
			# 2) Optionally unfreeze last 2 transformer blocks + final layer norm of the text encoder
			base = model.text_encoder.base_model
			if hasattr(base, "text_model"):
				encoder_layers = base.text_model.encoder.layers
				final_ln = base.text_model.final_layer_norm
			else:
				encoder_layers = base.encoder.layers
				final_ln = base.final_layer_norm

			# Unfreeze last two transformer layers
			for layer in encoder_layers[-2:]:
				for p in layer.parameters():
					p.requires_grad = True

			# Unfreeze final LayerNorm
			for p in final_ln.parameters():
				p.requires_grad = True

		return model

	def _encode_boards(self, board_seqs: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Encode variable-length board sequences and return padded embeddings plus mask."""
		lengths = [len(seq) for seq in board_seqs]
		if any(l == 0 for l in lengths):
			raise ValueError("Found an empty board sequence")

		flat_boards = torch.stack([b for seq in board_seqs for b in seq], dim=0).to(self.device)
		emb_list = []
		with torch.no_grad():
			for i in range(0, flat_boards.size(0), ENCODE_CHUNK_SIZE_BOARDS):
				chunk = flat_boards[i:i+ENCODE_CHUNK_SIZE_BOARDS]
				with amp.autocast('cuda', enabled=USE_AMP and torch.cuda.is_available()):
					chunk_emb = self.base_clip.board_encoder(chunk)
				# normalize in fp32 for numerical stability
				chunk_emb = F.normalize(chunk_emb.float(), dim=-1)
				emb_list.append(chunk_emb)
		board_emb_flat = torch.cat(emb_list, dim=0)

		split_embeds = torch.split(board_emb_flat, lengths)
		padded = pad_sequence(split_embeds, batch_first=True, padding_value=0.0)
		max_len = padded.size(1)
		pad_mask = torch.ones((len(board_seqs), max_len), device=self.device, dtype=torch.bool)
		for i, l in enumerate(lengths):
			pad_mask[i, :l] = False
		return padded, pad_mask

	def _encode_texts(self, text_seqs: List[List[str]]) -> torch.Tensor:
		"""Encode text sequences (one text per game) and return normalized embeddings."""
		lengths = [len(seq) for seq in text_seqs]
		if any(l == 0 for l in lengths):
			raise ValueError("Found an empty text sequence")

		flat_texts = [t for seq in text_seqs for t in seq]
		text_emb_chunks = []
		for i in range(0, len(flat_texts), ENCODE_CHUNK_SIZE_TEXTS):
			subtexts = flat_texts[i:i+ENCODE_CHUNK_SIZE_TEXTS]
			with amp.autocast('cuda', enabled=USE_AMP and torch.cuda.is_available()):
				sub_emb = self.base_clip.text_encoder(subtexts)
			# normalize in fp32
			sub_emb = F.normalize(sub_emb.float(), dim=-1)
			text_emb_chunks.append(sub_emb)
		text_emb_flat = torch.cat(text_emb_chunks, dim=0)

		split_embeds = torch.split(text_emb_flat, lengths)
		seq_embs = []
		for emb_seq in split_embeds:
			# Each sequence currently carries a single text; fall back to mean if more appear.
			if emb_seq.size(0) == 1:
				seq_embs.append(emb_seq[0])
			else:
				seq_embs.append(emb_seq.mean(dim=0))
		return torch.stack(seq_embs, dim=0)

	def forward(
		self,
		board_seqs: List[List[torch.Tensor]],
		text_seqs: List[List[str]],
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Produce normalized pooled embeddings for game boards and corresponding texts."""
		board_padded, board_mask = self._encode_boards(board_seqs)
		text_emb = self._encode_texts(text_seqs)

		board_pooled = self.board_pooler(board_padded, board_mask)
		return board_pooled, text_emb

	def encode_game(self, game_batch: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
		"""Encode a batch of game sequences using the board pooler."""
		B = game_batch.shape[0]
		board_seqs = []
		for i in range(B):
			seq = [game_batch[i, j] for j in range(lengths[i])]
			board_seqs.append(seq)
		board_padded, board_mask = self._encode_boards(board_seqs)
		board_pooled = self.board_pooler(board_padded, board_mask)
		return board_pooled


# =========================================================
# Loss and metrics
# =========================================================


def symmetric_clip_loss(game_emb: torch.Tensor, text_emb: torch.Tensor, temperature: float) -> torch.Tensor:
	"""Apply the bidirectional CLIP loss with a fixed temperature parameter."""
	game_emb = F.normalize(game_emb, dim=-1)
	text_emb = F.normalize(text_emb, dim=-1)
	logits = (game_emb @ text_emb.t()) / temperature
	targets = torch.arange(game_emb.size(0), device=game_emb.device)
	loss_i = F.cross_entropy(logits, targets)
	loss_t = F.cross_entropy(logits.t(), targets)
	return 0.5 * (loss_i + loss_t)


@torch.no_grad()
def compute_recall_at_k(game_emb: torch.Tensor, text_emb: torch.Tensor, k: int = 10) -> float:
	"""Compute recall@k for retrieving the matching game given each text query."""
	game_emb = F.normalize(game_emb, dim=-1)
	text_emb = F.normalize(text_emb, dim=-1)
	sims = text_emb @ game_emb.t()
	topk = sims.topk(k=min(k, sims.size(1)), dim=1).indices
	targets = torch.arange(game_emb.size(0), device=game_emb.device).unsqueeze(1)
	hits = (topk == targets).any(dim=1).float()
	return hits.mean().item()


def compute_ranking_metrics(
	game_emb: torch.Tensor,
	text_emb: torch.Tensor,
) -> Tuple[float, float]:
	"""Compute MRR and mean rank for text->game retrieval."""
	game_norm = F.normalize(game_emb, dim=-1)
	text_norm = F.normalize(text_emb, dim=-1)
	sims = text_norm @ game_norm.t()
	sorted_idx = torch.argsort(sims, dim=1, descending=True)
	targets = torch.arange(game_emb.size(0), device=game_emb.device)
	ranks = torch.full((text_emb.size(0),), fill_value=sims.size(1) + 1, dtype=torch.float32, device=text_emb.device)
	for i in range(sorted_idx.size(0)):
		pos = (sorted_idx[i] == targets[i]).nonzero(as_tuple=False)
		if pos.numel() > 0:
			ranks[i] = pos[0, 0].float() + 1.0  # convert to 1-based rank

	mrr = torch.mean(1.0 / ranks)
	mean_rank = torch.mean(ranks)
	return mrr.item(), mean_rank.item()


# =========================================================
# Training helpers
# =========================================================



def create_dataloaders(
	train_df: pd.DataFrame,
	val_df: Optional[pd.DataFrame],
	batch_size: int,
	val_size: float,
	num_workers: int,
	device: str,
	val_count: Optional[int] = None,
	group_col: Optional[str] = GROUP_COL,
) -> Tuple[DataLoader, DataLoader]:
	"""Split the dataframe into train/val loaders while preserving grouped games."""
	if val_df is None:
		if val_count is not None and val_count > 0:
			if len(train_df) <= val_count:
				val_df = train_df.copy()
				train_df = train_df.iloc[0:0]
			else:
				val_df = train_df.sample(n=val_count, random_state=42)
				train_df = train_df.drop(val_df.index).reset_index(drop=True)
				val_df = val_df.reset_index(drop=True)
		else:
			from sklearn.model_selection import train_test_split

			train_df, val_df = train_test_split(
				train_df,
				test_size=val_size,
				random_state=42,
				shuffle=True,
			)
	else:
		train_df = train_df.reset_index(drop=True)
		val_df = val_df.reset_index(drop=True)

	train_ds = ChessGamesDataset(train_df, board_seq_col=BOARD_SEQ_COL, text_seq_col=TEXT_SEQ_COL, group_col=group_col)
	val_ds = ChessGamesDataset(val_df, board_seq_col=BOARD_SEQ_COL, text_seq_col=TEXT_SEQ_COL, group_col=group_col)

	pin_memory = device.startswith("cuda")

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=True,
		collate_fn=collate_games,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=False,
		collate_fn=collate_games,
	)
	return train_loader, val_loader


def train_one_epoch(
	model: ChessGamesCLIP,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	device: str,
) -> float:
	"""Run a single training epoch over game sequences with optional AMP."""
	model.train()
	total_loss = 0.0
	steps = 0
	scaler = amp.GradScaler('cuda', enabled=USE_AMP and device.startswith("cuda"))
	optimizer.zero_grad()
	for board_seqs, text_seqs, _ in tqdm(loader, desc="Train", leave=False):
		with amp.autocast('cuda', enabled=USE_AMP and device.startswith("cuda")):
			game_emb, text_emb = model(board_seqs, text_seqs)
			loss = symmetric_clip_loss(game_emb, text_emb, TEMPERATURE)
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
		optimizer.zero_grad()
		total_loss += loss.item()
		steps += 1
	return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(model: ChessGamesCLIP, loader: DataLoader, device: str) -> Tuple[float, float, float, float, float, float, float]:
	"""Evaluate loss plus retrieval metrics on the validation split."""
	model.eval()
	total_loss = 0.0
	steps = 0
	all_game_emb = []
	all_text_emb = []

	for board_seqs, text_seqs, _ in tqdm(loader, desc="Val", leave=False):
		with amp.autocast('cuda', enabled=USE_AMP and device.startswith("cuda")):
			game_emb, text_emb = model(board_seqs, text_seqs)
			loss = symmetric_clip_loss(game_emb, text_emb, TEMPERATURE)
		total_loss += loss.item()
		steps += 1
		all_game_emb.append(game_emb.cpu())
		all_text_emb.append(text_emb.cpu())

	avg_loss = total_loss / max(steps, 1)
	game_all = torch.cat(all_game_emb, dim=0)
	text_all = torch.cat(all_text_emb, dim=0)
	r1 = compute_recall_at_k(game_all, text_all, k=1)
	r5 = compute_recall_at_k(game_all, text_all, k=5)
	r10 = compute_recall_at_k(game_all, text_all, k=10)
	mrr, mean_rank = compute_ranking_metrics(game_all, text_all)
	return avg_loss, r1, r5, r10, mrr, mean_rank


# =========================================================
# Public entrypoint
# =========================================================


def train_games(train_df: pd.DataFrame, val_df: pd.DataFrame) -> str:
	"""Train game-text alignment using frozen base encoders and poolers."""

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
	if device == "cuda":
		print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

	os.makedirs(OUTPUT_DIR, exist_ok=True)

	# Convert the flat position datasets into game-level sequences
	train_games_df = build_game_dataframe_from_positions(train_df)
	val_games_df = build_game_dataframe_from_positions(val_df)

	# Basic filtering: drop empty sequences
	train_games_df = train_games_df[(train_games_df[BOARD_SEQ_COL].apply(lambda x: len(x) > 0)) & (train_games_df[TEXT_SEQ_COL].apply(lambda x: len(x) > 0))]
	train_games_df = train_games_df.reset_index(drop=True)
	val_games_df = val_games_df[(val_games_df[BOARD_SEQ_COL].apply(lambda x: len(x) > 0)) & (val_games_df[TEXT_SEQ_COL].apply(lambda x: len(x) > 0))]
	val_games_df = val_games_df.reset_index(drop=True)
	print(f"Rows after grouping (train) and filtering by MIN_GAME_LEN={MIN_GAME_LEN}, MAX_GAME_LEN={MAX_GAME_LEN}: {len(train_games_df)}")
	print(f"Rows after grouping (val) and filtering by MIN_GAME_LEN={MIN_GAME_LEN}, MAX_GAME_LEN={MAX_GAME_LEN}: {len(val_games_df)}")

	train_loader, val_loader = create_dataloaders(
			train_games_df,
			val_games_df,
		batch_size=BATCH_SIZE,
		val_size=VAL_SIZE,
		num_workers=NUM_WORKERS,
		device=device,
		val_count=VAL_COUNT,
		group_col=GROUP_COL,
	)

	print("Building model and loading frozen base encoders...")
	model = ChessGamesCLIP(base_ckpt=PRETRAINED_CLIP_CKPT, device=device)
	model.to(device)

	# Inform how many games will be used for training and validation
	print(f"Games for training: {len(train_loader.dataset)}")
	print(f"Games for validation: {len(val_loader.dataset)}")

	# Train only poolers and CLS tokens
	trainable_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

	best_val = float("inf")
	best_r10 = 0.0
	best_path = os.path.join(OUTPUT_DIR, f"{MODEL_PREFIX}_best.pt")
	patience_ctr = 0
	history: List[dict] = []

	for epoch in range(1, EPOCHS + 1):
		print(f"\nEpoch {epoch}/{EPOCHS}")
		train_loss = train_one_epoch(model, train_loader, optimizer, device)
		val_loss, r1, r5, r10, mrr, mean_rank = evaluate(model, val_loader, device)

		print(
			f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
			f"R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f} "
			f"MRR={mrr:.4f} MeanRank={mean_rank:.2f}"
		)

		history.append({
			"epoch": epoch,
			"train_loss": train_loss,
			"val_loss": val_loss,
			"recall_at_1": r1,
			"recall_at_5": r5,
			"recall_at_10": r10,
			"mrr": mrr,
			"mean_rank": mean_rank,
		})

		metrics_path = os.path.join(OUTPUT_DIR, METRICS_CSV_NAME)
		pd.DataFrame(history).to_csv(metrics_path, index=False)

		# Save periodic checkpoint
		if SAVE_EVERY > 0 and (epoch % SAVE_EVERY == 0):
			periodic_path = os.path.join(OUTPUT_DIR, f"{MODEL_PREFIX}_epoch_{epoch:04d}_{r10:.4f}.pt")
			torch.save({
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"val_loss": val_loss,
				"recall_at_1": r1,
				"recall_at_5": r5,
				"recall_at_10": r10,
				"mrr": mrr,
				"mean_rank": mean_rank,
				"config": {
					"PRETRAINED_CLIP_CKPT": PRETRAINED_CLIP_CKPT,
					"POOLER_D_MODEL": POOLER_D_MODEL,
					"POOLER_N_HEADS": POOLER_N_HEADS,
					"POOLER_NUM_LAYERS": POOLER_NUM_LAYERS,
					"POOLER_DIM_FF": POOLER_DIM_FF,
					"POOLER_DROPOUT": POOLER_DROPOUT,
					"USE_LAYER_NORM": USE_LAYER_NORM,
					"TEMPERATURE": TEMPERATURE,
					"LR": LR,
					"WEIGHT_DECAY": WEIGHT_DECAY,
					"BATCH_SIZE": BATCH_SIZE,
					"EPOCHS": EPOCHS,
					"VAL_SIZE": VAL_SIZE,
					"VAL_COUNT": VAL_COUNT,
				},
			}, periodic_path)
			print(f"  >> Periodic checkpoint saved to {periodic_path}")

		improved = r10 > best_r10
		if improved:
			best_val = val_loss
			best_r10 = r10
			patience_ctr = 0
			best_path = os.path.join(OUTPUT_DIR, f"{MODEL_PREFIX}_best_{r10:.4f}.pt")
			torch.save({
				"epoch": epoch,
				"model_state_dict": model.state_dict(),
				"optimizer_state_dict": optimizer.state_dict(),
				"val_loss": val_loss,
				"recall_at_1": r1,
				"recall_at_5": r5,
				"recall_at_10": r10,
				"best_val_loss": best_val,
				"best_recall": best_r10,
				"mrr": mrr,
				"mean_rank": mean_rank,
				"config": {
					"PRETRAINED_CLIP_CKPT": PRETRAINED_CLIP_CKPT,
					"POOLER_D_MODEL": POOLER_D_MODEL,
					"POOLER_N_HEADS": POOLER_N_HEADS,
					"POOLER_NUM_LAYERS": POOLER_NUM_LAYERS,
					"POOLER_DIM_FF": POOLER_DIM_FF,
					"POOLER_DROPOUT": POOLER_DROPOUT,
					"USE_LAYER_NORM": USE_LAYER_NORM,
					"TEMPERATURE": TEMPERATURE,
					"LR": LR,
					"WEIGHT_DECAY": WEIGHT_DECAY,
					"BATCH_SIZE": BATCH_SIZE,
					"EPOCHS": EPOCHS,
					"VAL_SIZE": VAL_SIZE,
					"VAL_COUNT": VAL_COUNT,
				},
			}, best_path)
			print(f"  >> New best model saved to {best_path}")
		else:
			patience_ctr += 1

		if patience_ctr >= PATIENCE:
			print(f"Early stopping at epoch {epoch}: no improvement in R@10 for {PATIENCE} epochs.")
			break

	print("\nTraining finished.")
	print(f"Best val_loss: {best_val:.4f}, best R@10: {best_r10:.4f}")
	print(f"Best checkpoint path: {best_path}")
	return best_path


if __name__ == "__main__":
	train_df = pd.read_csv(TRAIN_CSV_PATH)
	val_df = pd.read_csv(VAL_CSV_PATH)
	train_games(train_df, val_df)
