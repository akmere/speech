from collections.abc import Callable
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


_WARNED_UNNORMALIZED_EMBEDDINGS = False


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Batch-hard triplet loss.

    embeddings: (B, D) normalized embeddings
    labels: (B,) int labels
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be (B, D)")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be (B,)")

    global _WARNED_UNNORMALIZED_EMBEDDINGS
    if not _WARNED_UNNORMALIZED_EMBEDDINGS and embeddings.numel() != 0:
        norms = embeddings.detach().norm(p=2, dim=1)
        if not torch.isfinite(norms).all():
            raise ValueError("embeddings contain NaN/Inf norms")
        max_dev = (norms - 1.0).abs().max().item()
        if max_dev > 1e-2:
            warnings.warn(
                "batch_hard_triplet_loss expects L2-normalized embeddings; "
                f"max |norm-1|={max_dev:.3g}. "
                "Enable l2_normalize in Audio2WordVectorEncoder or normalize before calling loss.",
                RuntimeWarning,
            )
            _WARNED_UNNORMALIZED_EMBEDDINGS = True

    dist = torch.cdist(embeddings, embeddings, p=2)
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


class Audio2WordVectorEncoder(nn.Module):
    """2-layer unidirectional GRU encoder producing one embedding per utterance."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        dropout: float = 0.3,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.l2_normalize = l2_normalize

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


@torch.no_grad()
def classifier(
    x: torch.Tensor,
    *,
    f: Callable[[torch.Tensor], torch.Tensor],
    prototypes: torch.Tensor,
    t: float,
    d: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> int:
    """Classifier with rejection option."""
    if prototypes.ndim != 2:
        raise ValueError("prototypes must be (C, D)")

    z = f(x)
    if z.ndim == 2 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 1:
        raise ValueError("f(x) must be a 1D embedding (D,) (or (1, D))")
    if prototypes.shape[1] != z.shape[0]:
        raise ValueError(
            f"prototype dim D={prototypes.shape[1]} must match embedding dim D={z.shape[0]}"
        )

    z = z.to(device=prototypes.device)

    if d is None:
        distances = torch.linalg.vector_norm(prototypes - z.unsqueeze(0), ord=2, dim=1)
    else:
        distances = d(prototypes, z)

    distances = distances.reshape(-1)
    if distances.numel() != prototypes.shape[0]:
        raise ValueError("distance function must return one distance per prototype")

    min_dist, argmin = torch.min(distances, dim=0)
    return int(argmin.item()) if float(min_dist.item()) < float(t) else -1


def load_model(model_path: str, device: torch.device) -> Audio2WordVectorEncoder:
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint["input_dim"]
    embedding_dim = checkpoint["embedding_dim"]
    l2_normalize = bool(checkpoint.get("l2_normalize", True))
    model = Audio2WordVectorEncoder(input_dim, embedding_dim, l2_normalize=l2_normalize)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def save_model(model: Audio2WordVectorEncoder, model_path: str):
    checkpoint = {
        "input_dim": model.gru.input_size,
        "embedding_dim": model.embedding_dim,
        "l2_normalize": bool(model.l2_normalize),
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, model_path)
