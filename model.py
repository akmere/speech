from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning as L
import wandb
from lightning.pytorch.loggers import WandbLogger
from dataset import (
    DatasetInfo,
    extract_dataset_word_filename,
    extract_or_cache_mfcc,
    get_keyword_embeddings,
    draw_embeddings,
)

from matplotlib import pyplot as plt
from matplotlib import ticker as mticker

try:
    # Optional: enables "true" DET axes (normal deviate / probit)
    from scipy.stats import norm as _norm
except Exception:
    _norm = None


_DET_MIN_RATE = 1e-2
_DET_MAX_RATE = 1.0
_DET_TICKS = (1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1)


def _fmt_det_tick(x: float, _pos: int) -> str:
    # Match the DET tick labels we display: 0.01 0.02 0.05 0.1 0.2 0.5
    if x >= 1.0:
        return "1.00"
    if x >= 0.1:
        return f"{x:.1f}"
    return f"{x:.2f}"


def style_det_axes(ax) -> None:
    """Style axes to look like classic DET (log-log on [0.01, 1]).

    Note: log scale cannot include 0, so we clamp the visible range to 0.01..1.
    """
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(_DET_MIN_RATE, _DET_MAX_RATE)
    ax.set_ylim(_DET_MIN_RATE, _DET_MAX_RATE)

    ax.xaxis.set_major_locator(mticker.FixedLocator(_DET_TICKS))
    ax.yaxis.set_major_locator(mticker.FixedLocator(_DET_TICKS))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_det_tick))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_det_tick))

    # Only show the major grid lines corresponding to the labeled ticks.
    ax.minorticks_off()
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(False, which="minor")


# Independently of how the system is configured, the prediction is
# obtained by selecting a keyword whose embedding is the closest
# to the input embedding and by using a threshold. The latter
# defines the minimum allowed distance to the closest keyword
# embedding. Precisely, the classification is defined as follows:
# c(x) =
# { argmin_i d(e_i, f(x)) if ∃j s.t. d(e_j, f(x)) < t
# −1 otherwise
# where c represents the classification function, x input audio to
# be classified, d is the distance metric, f is the encoder, ei the
# embedding of the i-th keyword, t the threshold and −1 indicates
# that no keyword embedding is close enough, i.e. no keyword
# has been detected
@dataclass
class KeywordEmbeddingIndex:
    """Prepacked keyword embeddings for fast nearest-neighbor classification.

    This avoids a Python loop by stacking keyword embeddings into a single
    tensor once (shape: (K, D)) and reusing it across many calls.
    """

    keywords: tuple[str, ...]
    embeddings: torch.Tensor  # (K, D)

    @classmethod
    def from_mapping(
        cls,
        keyword_embeddings: Mapping[str, torch.Tensor],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "KeywordEmbeddingIndex":
        keywords: list[str] = []
        vectors: list[torch.Tensor] = []
        for keyword, kw_emb in keyword_embeddings.items():
            if kw_emb.ndim != 1:
                raise ValueError(
                    f"keyword embedding for '{keyword}' must be 1D (D,), got shape={tuple(kw_emb.shape)}"
                )
            keywords.append(keyword)
            vectors.append(kw_emb)

        if not vectors:
            raise ValueError("keyword_embeddings is empty")

        emb = torch.stack(vectors, dim=0)
        if device is not None or dtype is not None:
            emb = emb.to(device=device, dtype=dtype)
        return cls(tuple(keywords), emb)


def classify_embedding(
    embedding: torch.Tensor,
    keyword_embeddings: Mapping[str, torch.Tensor] | KeywordEmbeddingIndex,
    threshold: float,
) -> str | None:
    """Classify an embedding by nearest keyword embedding under an L2 threshold.

    Fast path: pass a `KeywordEmbeddingIndex` to avoid stacking each call.
    """
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    if embedding.ndim != 1:
        raise ValueError(
            f"embedding must be 1D (D,), got shape={tuple(embedding.shape)}"
        )
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    if isinstance(keyword_embeddings, KeywordEmbeddingIndex):
        keywords = keyword_embeddings.keywords
        kw = keyword_embeddings.embeddings
    else:
        # Still vectorized, but will restack each call.
        keywords = tuple(keyword_embeddings.keys())
        if not keywords:
            return None
        kw = torch.stack([keyword_embeddings[k] for k in keywords], dim=0)

    if kw.ndim != 2:
        raise ValueError(
            f"keyword embeddings must be (K, D), got shape={tuple(kw.shape)}"
        )
    if kw.shape[1] != embedding.shape[0]:
        raise ValueError(
            f"dimension mismatch: embedding D={embedding.shape[0]} vs keyword D={kw.shape[1]}"
        )

    # Ensure embedding is on the same device/dtype as the keyword matrix.
    # (This is important for GPU inference. Prefer building the index on the
    # same device as the encoder to avoid per-call transfers.)
    embedding = embedding.to(device=kw.device, dtype=kw.dtype)

    # Compute squared L2 distances (avoid sqrt), then compare to threshold^2.
    # dist2[k] = ||kw[k] - embedding||_2^2
    diff = kw - embedding.unsqueeze(0)  # (K, D)
    dist2 = (diff * diff).sum(dim=1)  # (K,)
    best_idx = int(dist2.argmin().item())
    best_dist2 = float(dist2[best_idx].item())

    if best_dist2 < float(threshold) * float(threshold):
        return keywords[best_idx]
    return None


def calculate_missed_detection_and_false_positive_rates(
    model: "ConvStatsPoolEncoder | GRUEncoder",
    keyword_embedding_index: KeywordEmbeddingIndex,
    dataset_info: DatasetInfo,
    words: list[str],
    threshold: float,
) -> tuple[float, float]:
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    # Ensure the keyword embedding matrix lives on the same device as the model.
    if keyword_embedding_index.embeddings.device != model_device:
        keyword_embedding_index = KeywordEmbeddingIndex(
            keyword_embedding_index.keywords,
            keyword_embedding_index.embeddings.to(model_device),
        )

    total_samples: int = 0
    seen_words_samples: int = 0
    unseen_words_samples: int = 0
    missed_seen_predictions: int = 0
    false_positives: int = 0
    for word in words:
        samples = dataset_info.sample_word(word, n=100)
        for sample in samples:
            dataset, word, filename = extract_dataset_word_filename(sample)
            mfcc = extract_or_cache_mfcc(dataset, word, filename).to(model_device)
            embedding = model(mfcc.unsqueeze(0)).squeeze(0)
            predicted_word = classify_embedding(
                embedding, keyword_embedding_index, threshold=threshold
            )
            total_samples += 1
            if word in dataset_info.seen_words:
                if predicted_word != word:
                    missed_seen_predictions += 1
                seen_words_samples += 1
            else:
                if predicted_word is not None:
                    false_positives += 1
                unseen_words_samples += 1
    if total_samples == 0:
        return 0.0, 0.0
    return (
        missed_seen_predictions / seen_words_samples if seen_words_samples > 0 else 0.0,
        false_positives / unseen_words_samples if unseen_words_samples > 0 else 0.0,
    )


@torch.no_grad()
def det_points_for_thresholds_split(
    model: "ConvStatsPoolEncoder | GRUEncoder",
    keyword_embedding_index: KeywordEmbeddingIndex,
    dataset_info: DatasetInfo,
    *,
    target_words: list[str],
    non_target_words: list[str],
    n_per_word: int = 100,
    n_thresholds: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DET sweep for an explicit (target vs non-target) split.

    MDR: fraction of target samples that are missed (wrong nearest keyword OR dist >= thr).
    FPR: fraction of non-target samples that produce a false alarm (dist < thr).
    Threshold is in L2 distance units (same convention as classify_embedding()).
    """
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    kw = keyword_embedding_index.embeddings.to(model_device)
    keywords = keyword_embedding_index.keywords
    word_to_idx = {w: i for i, w in enumerate(keywords)}

    # Collect nearest distances for both target + non-target samples
    target_dists: list[float] = []
    target_correct: list[bool] = []
    non_target_dists: list[float] = []

    # Target (positive) samples
    for w in target_words:
        for sample in dataset_info.sample_word(w, n=n_per_word):
            dataset, word, filename = extract_dataset_word_filename(sample)
            mfcc = extract_or_cache_mfcc(dataset, word, filename).to(model_device)
            emb = model(mfcc.unsqueeze(0)).squeeze(0)

            diff = kw - emb.unsqueeze(0)
            dist2 = (diff * diff).sum(dim=1)
            best_idx = int(dist2.argmin().item())
            best_dist = float(torch.sqrt(dist2[best_idx]).item())

            target_dists.append(best_dist)
            target_correct.append(word_to_idx.get(word, -1) == best_idx)

    # Non-target (negative) samples
    for w in non_target_words:
        for sample in dataset_info.sample_word(w, n=n_per_word):
            dataset, word, filename = extract_dataset_word_filename(sample)
            mfcc = extract_or_cache_mfcc(dataset, word, filename).to(model_device)
            emb = model(mfcc.unsqueeze(0)).squeeze(0)

            diff = kw - emb.unsqueeze(0)
            dist2 = (diff * diff).sum(dim=1)
            best_idx = int(dist2.argmin().item())
            best_dist = float(torch.sqrt(dist2[best_idx]).item())

            non_target_dists.append(best_dist)

    if not target_dists and not non_target_dists:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty

    td = (
        np.asarray(target_dists, dtype=np.float64)
        if target_dists
        else np.asarray([], dtype=np.float64)
    )
    tc = (
        np.asarray(target_correct, dtype=bool)
        if target_correct
        else np.asarray([], dtype=bool)
    )
    nd = (
        np.asarray(non_target_dists, dtype=np.float64)
        if non_target_dists
        else np.asarray([], dtype=np.float64)
    )

    # Use a shared threshold range across both sets so the curve is well-defined.
    # Sweep beyond the observed max distance to reach more extreme operating points.
    all_d = (
        np.concatenate([td, nd]) if (td.size and nd.size) else (td if td.size else nd)
    )
    observed_hi = float(all_d.max()) if all_d.size else 1.0
    lo = 0.0
    hi = observed_hi * 2.0
    if not np.isfinite(lo) or not np.isfinite(hi):
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty
    if hi <= lo:
        hi = lo + 1e-6
    print(f"Threshold range: lo={lo}, hi={hi}")

    thresholds = np.linspace(lo, hi, int(n_thresholds), endpoint=True, dtype=np.float64)

    # MDR over targets
    if td.size:
        missed = (~tc)[None, :] | (td[None, :] >= thresholds[:, None])
        mdr = missed.mean(axis=1)
    else:
        mdr = np.zeros_like(thresholds)

    # FPR over non-targets
    if nd.size:
        fpr = (nd[None, :] < thresholds[:, None]).mean(axis=1)
    else:
        fpr = np.zeros_like(thresholds)

    return thresholds, mdr, fpr


@torch.no_grad()
def det_points_for_thresholds_split_avg_per_keyword(
    model: "ConvStatsPoolEncoder | GRUEncoder",
    keyword_embedding_index: KeywordEmbeddingIndex,
    dataset_info: DatasetInfo,
    *,
    target_words: list[str],
    non_target_words: list[str],
    n_per_word: int = 100,
    n_thresholds: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DET sweep computed per keyword then macro-averaged across keywords.

    This matches the common isolated-word KWS reporting described in the paper:
    compute a DET curve for each keyword, then average to get a single curve.

    Per keyword w:
      - positives: samples of w
      - negatives: samples of non_target_words
      - miss if (nearest keyword != w) OR (dist >= thr)
      - false alarm if (nearest keyword == w) AND (dist < thr) on negatives

    Threshold is in L2 distance units (same convention as classify_embedding()).
    """

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    kw = keyword_embedding_index.embeddings.to(model_device)
    keywords = keyword_embedding_index.keywords
    word_to_kw_idx = {w: i for i, w in enumerate(keywords)}

    non_target_set = set(non_target_words)

    # Evaluate the union of the provided word lists (avoid duplicate sampling).
    eval_words: list[str] = []
    seen: set[str] = set()
    for w in list(target_words) + list(non_target_words):
        if w not in seen:
            eval_words.append(w)
            seen.add(w)

    true_word_list: list[str] = []
    best_idx_list: list[int] = []
    best_dist_list: list[float] = []
    is_non_target_list: list[bool] = []

    for w in eval_words:
        for sample in dataset_info.sample_word(w, n=n_per_word):
            dataset, word, filename = extract_dataset_word_filename(sample)
            mfcc = extract_or_cache_mfcc(dataset, word, filename).to(model_device)
            emb = model(mfcc.unsqueeze(0)).squeeze(0)

            diff = kw - emb.unsqueeze(0)
            dist2 = (diff * diff).sum(dim=1)
            bi = int(dist2.argmin().item())
            bd = float(torch.sqrt(dist2[bi]).item())

            true_word_list.append(word)
            best_idx_list.append(bi)
            best_dist_list.append(bd)
            is_non_target_list.append(word in non_target_set)

    if not best_dist_list:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty

    true_words = np.asarray(true_word_list, dtype=object)
    pred_idx = np.asarray(best_idx_list, dtype=np.int64)
    dist = np.asarray(best_dist_list, dtype=np.float64)
    is_non_target = np.asarray(is_non_target_list, dtype=bool)

    # Shared threshold grid (needed to average curves keyword-wise).
    # Sweep beyond the observed max distance to reach more extreme operating points.
    observed_hi = float(np.nanmax(dist))
    lo = 0.0
    hi = observed_hi * 2.0
    if not np.isfinite(lo) or not np.isfinite(hi):
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty
    if hi <= lo:
        hi = lo + 1e-6
    print(f"Threshold range: lo={lo}, hi={hi}")
    thresholds = np.linspace(lo, hi, int(n_thresholds), endpoint=True, dtype=np.float64)

    # Negatives are shared for all per-keyword detectors in this split.
    neg_mask = is_non_target
    neg_dist = dist[neg_mask]
    neg_pred = pred_idx[neg_mask]
    n_neg = int(neg_dist.shape[0])
    if n_neg == 0:
        # Degenerate: no negatives. Keep shape consistent.
        mdr = np.zeros_like(thresholds)
        return thresholds, mdr, np.zeros_like(thresholds)

    sum_mdr = np.zeros_like(thresholds, dtype=np.float64)
    sum_fpr = np.zeros_like(thresholds, dtype=np.float64)
    used = 0

    for w in target_words:
        kw_idx = word_to_kw_idx.get(w)
        if kw_idx is None:
            continue

        pos_mask = true_words == w
        pos_dist = dist[pos_mask]
        pos_pred = pred_idx[pos_mask]
        n_pos = int(pos_dist.shape[0])
        if n_pos == 0:
            continue

        # MDR_w(thr): misses = wrong_nn + correct_nn_with_dist>=thr
        correct_mask = pos_pred == kw_idx
        correct_d = np.sort(pos_dist[correct_mask])
        n_correct = int(correct_d.shape[0])
        n_wrong = n_pos - n_correct

        # Count correct detections with dist < thr
        correct_lt = np.searchsorted(correct_d, thresholds, side="left")
        misses = n_wrong + (n_correct - correct_lt)
        mdr_w = misses.astype(np.float64) / float(n_pos)

        # FPR_w(thr): negatives classified as w with dist < thr
        neg_match_d = np.sort(neg_dist[neg_pred == kw_idx])
        fp = np.searchsorted(neg_match_d, thresholds, side="left")
        fpr_w = fp.astype(np.float64) / float(n_neg)

        sum_mdr += mdr_w
        sum_fpr += fpr_w
        used += 1

    if used == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty

    return thresholds, (sum_mdr / float(used)), (sum_fpr / float(used))


@torch.no_grad()
def det_points_for_thresholds(
    model: "ConvStatsPoolEncoder | GRUEncoder",
    keyword_embedding_index: KeywordEmbeddingIndex,
    dataset_info: DatasetInfo,
    words: list[str],
    *,
    n_per_word: int = 100,
    n_thresholds: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (thresholds, mdr, fpr) for a sweep of thresholds.

    MDR is computed over seen-word samples, FPR over unseen-word samples.
    Threshold is in the same units as the L2 distance used by `classify_embedding`.
    """

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    kw = keyword_embedding_index.embeddings.to(model_device)
    keywords = keyword_embedding_index.keywords
    word_to_idx = {w: i for i, w in enumerate(keywords)}

    dists: list[float] = []
    is_seen: list[bool] = []
    nearest_correct: list[bool] = []

    for word in words:
        samples = dataset_info.sample_word(word, n=n_per_word)
        for sample in samples:
            dataset, word, filename = extract_dataset_word_filename(sample)
            mfcc = extract_or_cache_mfcc(dataset, word, filename).to(model_device)
            emb = model(mfcc.unsqueeze(0)).squeeze(0)

            diff = kw - emb.unsqueeze(0)
            dist2 = (diff * diff).sum(dim=1)
            best_idx = int(dist2.argmin().item())
            best_dist = float(torch.sqrt(dist2[best_idx]).item())

            dists.append(best_dist)
            if word in dataset_info.seen_words:
                is_seen.append(True)
                nearest_correct.append(word_to_idx.get(word, -1) == best_idx)
            else:
                is_seen.append(False)
                nearest_correct.append(False)

    if not dists:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty

    dists_np = np.asarray(dists, dtype=np.float64)
    is_seen_np = np.asarray(is_seen, dtype=bool)
    nearest_correct_np = np.asarray(nearest_correct, dtype=bool)

    # Thresholds in distance-space.
    lo = float(dists_np.min())
    hi = float(dists_np.max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty
    if hi <= lo:
        hi = lo + 1e-6
    thresholds = np.linspace(
        lo, hi, int(n_thresholds), endpoint=True, retstep=False, dtype=np.float64
    )

    seen_d = dists_np[is_seen_np]
    seen_correct = nearest_correct_np[is_seen_np]
    unseen_d = dists_np[~is_seen_np]

    if seen_d.size:
        # Miss if nearest keyword is wrong OR distance >= threshold.
        missed = (~seen_correct)[None, :] | (seen_d[None, :] >= thresholds[:, None])
        mdr = missed.mean(axis=1)
    else:
        mdr = np.zeros_like(thresholds)

    if unseen_d.size:
        # False positive if an unseen word is within threshold of any keyword.
        fpr = (unseen_d[None, :] < thresholds[:, None]).mean(axis=1)
    else:
        fpr = np.zeros_like(thresholds)

    return thresholds, mdr, fpr


def auc_from_det_points(
    fprs: np.ndarray, mdrs: np.ndarray, *, add_endpoints: bool = True
) -> tuple[float, float]:
    """Return (auc_roc, auc_det) from arrays of fpr and mdr.

    auc_roc: area under ROC (TPR vs FPR), higher is better.
    auc_det: area under DET (MDR vs FPR), lower is better.
    """
    if fprs.size == 0 or mdrs.size == 0:
        return 0.0, 0.0

    f = np.asarray(fprs, dtype=np.float64)
    m = np.asarray(mdrs, dtype=np.float64)

    # Sort by FPR for integration
    order = np.argsort(f)
    f = f[order]
    m = m[order]

    # Clamp to valid ranges (rates)
    f = np.clip(f, 0.0, 1.0)
    m = np.clip(m, 0.0, 1.0)
    tpr = 1.0 - m

    if add_endpoints:
        # Standard ROC endpoints
        f = np.concatenate(([0.0], f, [1.0]))
        tpr = np.concatenate(([0.0], tpr, [1.0]))
        m = np.concatenate(([1.0], m, [0.0]))  # DET endpoints consistent with above

    auc_roc = float(np.trapz(tpr, f))
    auc_det = float(np.trapz(m, f))
    return auc_roc, auc_det


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Batch-hard triplet loss.

    embeddings: (B, D) normalized embeddings
    labels: (B,) int labels
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be (B, D)")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be (B,)")

    x = embeddings
    x2 = (x * x).sum(dim=1, keepdim=True)  # (B,1)
    dist2 = (x2 + x2.T - 2.0 * (x @ x.T)).clamp_min(0.0)  # (B,B)
    dist = dist2  # use squared distances
    B = dist.shape[0]

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(B, device=labels.device, dtype=torch.bool)

    pos_mask = labels_eq & ~eye
    neg_mask = ~labels_eq

    pos_dist = dist.masked_fill(~pos_mask, float("-inf"))
    hardest_pos = pos_dist.max(dim=1).values

    neg_dist = dist.masked_fill(~neg_mask, float("inf"))
    hardest_neg = neg_dist.min(dim=1).values

    loss = F.relu(hardest_pos - hardest_neg + margin)
    loss = loss[torch.isfinite(loss)]
    return loss.mean() if loss.numel() else torch.tensor(0.0, device=embeddings.device)


def batch_hard_triplet_loss_cosine(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Batch-hard triplet loss using cosine similarity.

    Assumes embeddings are L2-normalized.

    We define similarity s(i,j) = <e_i, e_j>.
    Hardest positive: least similar positive.
    Hardest negative: most similar negative.

    Loss per anchor: relu(hardest_neg - hardest_pos + margin)
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be (B, D)")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be (B,)")

    # Cosine similarity matrix: (B,B)
    sim = embeddings @ embeddings.T
    B = sim.shape[0]

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(B, device=labels.device, dtype=torch.bool)

    pos_mask = labels_eq & ~eye
    neg_mask = ~labels_eq

    # Hardest positive: minimum similarity among positives.
    pos_sim = sim.masked_fill(~pos_mask, float("inf"))
    hardest_pos = pos_sim.min(dim=1).values

    # Hardest negative: maximum similarity among negatives.
    neg_sim = sim.masked_fill(~neg_mask, float("-inf"))
    hardest_neg = neg_sim.max(dim=1).values

    loss = F.relu(hardest_neg - hardest_pos + margin)
    loss = loss[torch.isfinite(loss)]
    return loss.mean() if loss.numel() else torch.tensor(0.0, device=embeddings.device)


class GRUEncoder(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        lr: float,
        margin: float,
        embedding_dim: int,
        dropout: float,
        threshold: float,
        dataset_info: DatasetInfo,
        det_curves: bool,
        embeddings_words: list[str],
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.margin = margin
        self.input_dim = int(input_dim)
        self.embedding_dim = embedding_dim
        self.l2_normalize = l2_normalize
        self.threshold = threshold
        self.dataset_info = dataset_info
        self.det_curves = det_curves
        self.embeddings_words = embeddings_words

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=embedding_dim,
            num_layers=2,
            dropout=dropout,
            bidirectional=False,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        if lengths is not None:
            if lengths.dtype != torch.int64:
                lengths = lengths.to(torch.int64)
            # sanity: lengths must be within [1, T]
            T = x.shape[1]
            if int(lengths.min().item()) < 1 or int(lengths.max().item()) > T:
                raise ValueError(
                    f"Bad lengths: min={int(lengths.min())} max={int(lengths.max())} T={T}"
                )

            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)

        emb = h_n[-1]
        if self.l2_normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    def _common_step(self, batch, batch_idx):
        x, lengths, y = batch  # x: (B, T, input_dim)
        embeddings = self.forward(x, lengths)  # (B, embedding_dim)
        loss = batch_hard_triplet_loss(embeddings, y, self.margin)
        return loss, embeddings, y

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.print(
                f"epoch={epoch} train_loss={train_loss.detach().cpu().item():.6f}"
            )

    def on_validation_epoch_end(self):
        epoch = self.current_epoch
        validation_loss = self.trainer.callback_metrics.get("val_loss")
        if validation_loss is not None:
            self.print(
                f"epoch={epoch} validation_loss={validation_loss.detach().cpu().item():.6f}"
            )

        # Then,
        # we compute a DET curve for each word in the sets and average
        # them to obtain one DET curve for the seen keywords and
        # one for the unseen ones

        # keyword_embeddings = get_keyword_embeddings(
        #     self, self.dataset_info.seen_words, self.dataset_info
        # )
        # keyword_embedding_index = KeywordEmbeddingIndex.from_mapping(
        #     keyword_embeddings, device=self.device
        # )
        # mdr, fpr = calculate_missed_detection_and_false_positive_rates(
        #     self,
        #     keyword_embedding_index,
        #     self.dataset_info,
        #     self.dataset_info.seen_words + self.dataset_info.unseen_words,
        #     self.threshold,
        # )

        # self.log_dict(
        #     {
        #         "missed_detection_rate": mdr,
        #         "false_positive_rate": fpr,
        #     }
        # )
        if self.det_curves:
            os.makedirs("plots", exist_ok=True)

            seen_kw = get_keyword_embeddings(
                self, self.dataset_info.seen_words, self.dataset_info
            )
            unseen_kw = get_keyword_embeddings(
                self, self.dataset_info.unseen_words, self.dataset_info
            )
            seen_index = KeywordEmbeddingIndex.from_mapping(seen_kw, device=self.device)
            unseen_index = KeywordEmbeddingIndex.from_mapping(
                unseen_kw, device=self.device
            )

            # Initialize AUC values so they are always bound (helps static type checkers).
            auc_roc_s = float("nan")
            auc_det_s = float("nan")
            auc_roc_u = float("nan")
            auc_det_u = float("nan")

            # Seen: DET sweep using the same decision rule as `classify_embedding`
            # (nearest keyword + threshold).
            thr_s, mdr_s, fpr_s = det_points_for_thresholds_split(
                self,
                seen_index,
                self.dataset_info,
                target_words=self.dataset_info.seen_words,
                non_target_words=self.dataset_info.unseen_words,
            )
            if thr_s.size:
                auc_roc_s, auc_det_s = auc_from_det_points(fpr_s, mdr_s)
                self.log_dict(
                    {"auc_roc_seen": auc_roc_s, "auc_det_seen": auc_det_s},
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                )

            # Unseen: DET sweep using the same decision rule as `classify_embedding`.
            thr_u, mdr_u, fpr_u = det_points_for_thresholds_split(
                self,
                unseen_index,
                self.dataset_info,
                target_words=self.dataset_info.unseen_words,
                non_target_words=self.dataset_info.seen_words,
            )
            if thr_u.size:
                auc_roc_u, auc_det_u = auc_from_det_points(fpr_u, mdr_u)
                self.log_dict(
                    {"auc_roc_unseen": auc_roc_u, "auc_det_unseen": auc_det_u},
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                )

            # One plot: overlay both curves + legend
            if thr_s.size or thr_u.size:
                fig, ax = plt.subplots(figsize=(6, 6))

                # Avoid log(0) and match a classic DET view range.
                eps = _DET_MIN_RATE

                if thr_s.size:
                    ax.plot(
                        np.clip(fpr_s, eps, 1.0),
                        np.clip(mdr_s, eps, 1.0),
                        label=f"Seen targets (AUC_det={auc_det_s:.4f})",
                    )
                if thr_u.size:
                    ax.plot(
                        np.clip(fpr_u, eps, 1.0),
                        np.clip(mdr_u, eps, 1.0),
                        label=f"Unseen targets (AUC_det={auc_det_u:.4f})",
                    )

                style_det_axes(ax)
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("Missed detection rate")
                ax.set_title(f"DET curves (classifier, epoch {epoch})")
                ax.legend(loc="best")

                out_path = os.path.join(
                    "plots", f"det_seen_vs_unseen_epoch_{epoch}.png"
                )
                fig.savefig(out_path, bbox_inches="tight")
                plt.close(fig)

                if isinstance(self.logger, WandbLogger):
                    self.logger.log_image(
                        "det_curves_seen_vs_unseen",
                        [
                            wandb.Image(
                                out_path, caption=f"DET seen vs unseen epoch {epoch}"
                            )
                        ],
                    )

            embeddings_path = os.path.join("plots", f"embeddings_epoch_{epoch}.png")
            draw_embeddings(
                save_path=embeddings_path,
                dataset_info=self.dataset_info,
                model=self,
                words=self.embeddings_words,
            )
            if isinstance(self.logger, WandbLogger):
                self.logger.log_image(
                    "embeddings",
                    [
                        wandb.Image(
                            embeddings_path,
                            caption=f"Embeddings epoch {epoch}",
                        )
                    ],
                )

    def configure_optimizers(
        self,
    ):
        return optim.Adam(self.parameters(), lr=self.lr)


class ConvStatsPoolEncoder(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        lr: float,
        margin: float,
        channels: int,
        threshold: float,
        dataset_info: DatasetInfo,
        det_curves: bool,
        embeddings_words: list[str],
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.l2_normalize = l2_normalize
        self.lr = lr
        self.margin = margin
        self.threshold = threshold
        self.dataset_info = dataset_info
        self.det_curves = det_curves
        self.embeddings_words = embeddings_words

        self.net = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        # stats pooling produces (mean||std) => 2*channels
        self.proj = nn.Sequential(
            nn.Linear(2 * channels, 2 * channels),
            nn.ReLU(inplace=True),
            nn.Linear(2 * channels, embedding_dim),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)

        h = self.net(x)  # (B,C,T)
        B, C, T = h.shape

        if lengths is None:
            mean = h.mean(dim=2)
            std = h.std(dim=2, unbiased=False)
        else:
            if lengths.dtype != torch.int64:
                lengths = lengths.to(torch.int64)
            lengths = lengths.clamp(min=1, max=T)

            # mask: (B,T) True for valid frames
            t = torch.arange(T, device=h.device).unsqueeze(0)  # (1,T)
            mask = t < lengths.unsqueeze(1)  # (B,T)
            mask_f = mask.unsqueeze(1).to(h.dtype)  # (B,1,T)

            denom = mask_f.sum(dim=2).clamp_min(1.0)  # (B,1)
            mean = (h * mask_f).sum(dim=2) / denom  # (B,C)

            var = ((h - mean.unsqueeze(2)) ** 2 * mask_f).sum(dim=2) / denom
            std = torch.sqrt(var.clamp_min(1e-8))  # (B,C)

        pooled = torch.cat([mean, std], dim=1)  # (B,2C)
        emb = self.proj(pooled)  # (B,D)

        if self.l2_normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    def _common_step(self, batch, batch_idx):
        x, lengths, y = batch  # x: (B, T, input_dim)
        embeddings = self.forward(x, lengths)  # (B, embedding_dim)
        loss = batch_hard_triplet_loss(embeddings, y, self.margin)
        return loss, embeddings, y

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    # def predict_step(self, batch, batch_idx):
    #     loss, scores, y = self._common_step(batch, batch_idx)
    #     preds = torch.argmax(scores, dim=1)
    #     return preds

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.print(
                f"epoch={epoch} train_loss={train_loss.detach().cpu().item():.6f}"
            )

    def on_validation_epoch_end(self):
        epoch = self.current_epoch
        validation_loss = self.trainer.callback_metrics.get("val_loss")
        if validation_loss is not None:
            self.print(
                f"epoch={epoch} validation_loss={validation_loss.detach().cpu().item():.6f}"
            )

        # Then,
        # we compute a DET curve for each word in the sets and average
        # them to obtain one DET curve for the seen keywords and
        # one for the unseen ones

        # keyword_embeddings = get_keyword_embeddings(
        #     self, self.dataset_info.seen_words, self.dataset_info
        # )
        # keyword_embedding_index = KeywordEmbeddingIndex.from_mapping(
        #     keyword_embeddings, device=self.device
        # )
        # mdr, fpr = calculate_missed_detection_and_false_positive_rates(
        #     self,
        #     keyword_embedding_index,
        #     self.dataset_info,
        #     self.dataset_info.seen_words + self.dataset_info.unseen_words,
        #     self.threshold,
        # )

        # self.log_dict(
        #     {
        #         "missed_detection_rate": mdr,
        #         "false_positive_rate": fpr,
        #     }
        # )
        if self.det_curves:
            os.makedirs("plots", exist_ok=True)

            seen_kw = get_keyword_embeddings(
                self, self.dataset_info.seen_words, self.dataset_info
            )
            unseen_kw = get_keyword_embeddings(
                self, self.dataset_info.unseen_words, self.dataset_info
            )
            seen_index = KeywordEmbeddingIndex.from_mapping(seen_kw, device=self.device)
            unseen_index = KeywordEmbeddingIndex.from_mapping(
                unseen_kw, device=self.device
            )

            # Initialize AUC values so they are always bound (helps static type checkers).
            auc_roc_s = float("nan")
            auc_det_s = float("nan")
            auc_roc_u = float("nan")
            auc_det_u = float("nan")

            # Seen: DET sweep using the same decision rule as `classify_embedding`.
            thr_s, mdr_s, fpr_s = det_points_for_thresholds_split(
                self,
                seen_index,
                self.dataset_info,
                target_words=self.dataset_info.seen_words,
                non_target_words=self.dataset_info.unseen_words,
            )
            if thr_s.size:
                auc_roc_s, auc_det_s = auc_from_det_points(fpr_s, mdr_s)
                self.log_dict(
                    {"auc_roc_seen": auc_roc_s, "auc_det_seen": auc_det_s},
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                )

            # Unseen: DET sweep using the same decision rule as `classify_embedding`.
            thr_u, mdr_u, fpr_u = det_points_for_thresholds_split(
                self,
                unseen_index,
                self.dataset_info,
                target_words=self.dataset_info.unseen_words,
                non_target_words=self.dataset_info.seen_words,
            )
            if thr_u.size:
                auc_roc_u, auc_det_u = auc_from_det_points(fpr_u, mdr_u)
                self.log_dict(
                    {"auc_roc_unseen": auc_roc_u, "auc_det_unseen": auc_det_u},
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                )

            # One plot: overlay both curves + legend
            if thr_s.size or thr_u.size:
                fig, ax = plt.subplots(figsize=(6, 6))

                # Avoid log(0) and match a classic DET view range.
                eps = _DET_MIN_RATE

                if thr_s.size:
                    ax.plot(
                        np.clip(fpr_s, eps, 1.0),
                        np.clip(mdr_s, eps, 1.0),
                        label=f"Seen targets (AUC_det={auc_det_s:.4f})",
                    )
                if thr_u.size:
                    ax.plot(
                        np.clip(fpr_u, eps, 1.0),
                        np.clip(mdr_u, eps, 1.0),
                        label=f"Unseen targets (AUC_det={auc_det_u:.4f})",
                    )

                style_det_axes(ax)
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("Missed detection rate")
                ax.set_title(f"DET curves (classifier, epoch {epoch})")
                ax.legend(loc="best")

                out_path = os.path.join(
                    "plots", f"det_seen_vs_unseen_epoch_{epoch}.png"
                )
                fig.savefig(out_path, bbox_inches="tight")
                plt.close(fig)

                if isinstance(self.logger, WandbLogger):
                    self.logger.log_image(
                        "det_curves_seen_vs_unseen",
                        [
                            wandb.Image(
                                out_path, caption=f"DET seen vs unseen epoch {epoch}"
                            )
                        ],
                    )

            embeddings_path = os.path.join("plots", f"embeddings_epoch_{epoch}.png")
            draw_embeddings(
                save_path=embeddings_path,
                dataset_info=self.dataset_info,
                model=self,
                words=self.embeddings_words,
            )
            if isinstance(self.logger, WandbLogger):
                self.logger.log_image(
                    "embeddings",
                    [
                        wandb.Image(
                            embeddings_path,
                            caption=f"Embeddings epoch {epoch}",
                        )
                    ],
                )

    def configure_optimizers(
        self,
    ):
        return optim.Adam(self.parameters(), lr=self.lr)


def get_embeddings_for_word(
    model: L.LightningModule,
    word: str,
):
    pass
