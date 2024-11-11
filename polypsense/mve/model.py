import lightning as L
import torch
import torch.nn as nn

from polypsense.sfe.model import SimCLR, accuracy


class MultiViewEncoder(L.LightningModule):
    def __init__(
        self,
        sfe=None,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        lr=1e-4,
    ):
        super().__init__()

        self.sfe = sfe  # single frame encoder
        self.transformer_encoder = self._build_transformer_encoder(
            d_model,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # [1, 1, d]
        self.head = nn.Linear(d_model, 128)
        self.simclr = SimCLR(n_views=2, temperature=0.07)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

        # While lightning logger encourages to ignore `sfe`, we still need to
        # save its hparams in the ckeckpoint such that
        # `MultiViewEncoder.load_from_checkpoint` is able to automatically
        # instanciate the `SingleFrameEncoder` model and load its state dict.
        # Few workarounds have been posted in this issue:
        # https://github.com/Lightning-AI/pytorch-lightning/issues/11494 Since
        # we do not need the `hparams.yml` file we simple turn of logger and
        # save all hyperparameters in the checkpoint.
        self.save_hyperparameters(logger=False)

    def forward(self, x):
        # x: [b, s, c, h, w]
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)  # [b*s, c, h, w]
        x = self.sfe(x)  # [b*s, d]
        x = x.view(b, s, -1)  # [b, s, d]
        x = self._add_cls_token(x)  # [b, s+1, d]
        x = x.permute(1, 0, 2)  # [s+1, b, d]
        x = self.transformer_encoder(x)  # [s+1, b, d]
        x = self.head(x[0])  # [b, d]
        return x

    def step(self, batch, stage=None):
        z = self(batch)
        logits, labels = self.simclr(z)
        loss = self.criterion(logits, labels)

        if stage:
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc_top1", top1)
            self.log(f"{stage}_acc_top5", top5)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _add_cls_token(self, x):
        # x: [b, s, d]
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)  # [b, 1, d]
        return torch.cat((cls_tokens, x), dim=1)  # [b, s+1, d]

    def _build_transformer_encoder(
        self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout
    ):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
        )
        transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
        )
        return transformer_encoder
