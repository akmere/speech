from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning as L
from sklearn.metrics import det_curve

from dataset import (
    DatasetInfo,
    extract_dataset_word_filename,
    extract_or_cache_mfcc,
    get_keyword_embeddings,
)

from matplotlib import pyplot as plt

try:
    # Optional: enables "true" DET axes (normal deviate / probit)
    from scipy.stats import norm as _norm
except Exception:
    _norm = None


def calculate_det_curve(
    model: "ConvStatsPoolEncoder | GRUEncoder",
    keyword_embedding_index: KeywordEmbeddingIndex,
    dataset_info: DatasetInfo,
    words: list[str],
    thresholds: Iterable[float],
) -> tuple[list[float], list[float], list[float]]:
    """Returns (thresholds, mdrs, fprs)."""
    model.eval()
    thr_list: list[float] = []
    mdrs: list[float] = []
    fprs: list[float] = []
    with torch.no_grad():
        for t in thresholds:
            mdr, fpr = calculate_missed_detection_and_false_positive_rates(
                model,
                keyword_embedding_index,
                dataset_info,
                words,
                threshold=float(t),
            )
            thr_list.append(float(t))
            mdrs.append(float(mdr))
            fprs.append(float(fpr))
    return thr_list, mdrs, fprs


def plot_det_curve(
    fprs: list[float],
    mdrs: list[float],
    *,
    title: str = "DET curve",
    use_probit_axes: bool = True,
):
    """Returns a matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=140)

    # Clamp away from 0/1 to avoid infinities on probit scale
    eps = 1e-6
    fx = [min(max(x, eps), 1.0 - eps) for x in fprs]
    my = [min(max(y, eps), 1.0 - eps) for y in mdrs]

    if use_probit_axes and _norm is not None:
        x = _norm.ppf(fx)
        y = _norm.ppf(my)
        ax.plot(x, y, linewidth=2)
        ax.set_xlabel("False positive rate (probit)")
        ax.set_ylabel("Missed detection rate (probit)")
        ax.set_title(title + " (probit)")
    else:
        ax.plot(fx, my, linewidth=2)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("Missed detection rate")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


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
    total_samples: int = 0
    seen_words_samples: int = 0
    unseen_words_samples: int = 0
    missed_seen_predictions: int = 0
    false_positives: int = 0
    for word in words:
        samples = dataset_info.sample_word(word, n=100)
        for sample in samples:
            dataset, word, filename = extract_dataset_word_filename(sample)
            mfcc = extract_or_cache_mfcc(dataset, word, filename)
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
                if predicted_word == word:
                    false_positives += 1
                unseen_words_samples += 1
    if total_samples == 0:
        return 0.0, 0.0
    return (
        missed_seen_predictions / seen_words_samples if seen_words_samples > 0 else 0.0,
        false_positives / unseen_words_samples if unseen_words_samples > 0 else 0.0,
    )


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
        keyword_embeddings = get_keyword_embeddings(
            self, self.dataset_info.seen_words, self.dataset_info
        )
        keyword_embedding_index = KeywordEmbeddingIndex.from_mapping(keyword_embeddings)
        mdr, fpr = calculate_missed_detection_and_false_positive_rates(
            self,
            keyword_embedding_index,
            self.dataset_info,
            self.dataset_info.seen_words + self.dataset_info.unseen_words,
            self.threshold,
        )
        self.log_dict(
            {
                "missed_detection_rate": mdr,
                "false_positive_rate": fpr,
            }
        )
        if self.det_curves:
            thresholds = torch.linspace(0.0, 2.0, steps=60).tolist()

            thr, mdrs, fprs = calculate_det_curve(
                self,
                keyword_embedding_index,
                self.dataset_info,
                self.dataset_info.seen_words + self.dataset_info.unseen_words,
                thresholds,
            )
            fig = plot_det_curve(
                fprs,
                mdrs,
                title=f"DET (epoch {epoch})",
                use_probit_axes=True,
            )
            fig.savefig(f"plots/det_curve_epoch_{epoch}.png")
            plt.close(fig)

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
        keyword_embeddings = get_keyword_embeddings(
            self, self.dataset_info.seen_words, self.dataset_info
        )
        keyword_embedding_index = KeywordEmbeddingIndex.from_mapping(keyword_embeddings)
        mdr, fpr = calculate_missed_detection_and_false_positive_rates(
            self,
            keyword_embedding_index,
            self.dataset_info,
            self.dataset_info.seen_words + self.dataset_info.unseen_words,
            self.threshold,
        )
        self.log_dict(
            {
                "missed_detection_rate": mdr,
                "false_positive_rate": fpr,
            }
        )
        if self.det_curves:
            thresholds = torch.linspace(0.0, 2.0, steps=60).tolist()

            thr, mdrs, fprs = calculate_det_curve(
                self,
                keyword_embedding_index,
                self.dataset_info,
                self.dataset_info.seen_words + self.dataset_info.unseen_words,
                thresholds,
            )
            fig = plot_det_curve(
                fprs,
                mdrs,
                title=f"DET (epoch {epoch})",
                use_probit_axes=True,
            )
            fig.savefig(f"plots/det_curve_epoch_{epoch}.png")
            plt.close(fig)

    def configure_optimizers(
        self,
    ):
        return optim.Adam(self.parameters(), lr=self.lr)


def get_embeddings_for_word(
    model: L.LightningModule,
    word: str,
):
    pass
