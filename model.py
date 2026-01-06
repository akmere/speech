import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning as L


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


class Model(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        lr: float,
        embedding_dim: int = 64,
        dropout: float = 0.3,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.lr = lr
        self.input_dim = int(input_dim)
        self.embedding_dim = embedding_dim
        self.l2_normalize = l2_normalize
        # self.loss_fn = nn.CrossEntropyLoss()

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
        loss = batch_hard_triplet_loss(embeddings, y, 1.0)
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

    def configure_optimizers(
        self,
    ):
        return optim.Adam(self.parameters(), lr=self.lr)
