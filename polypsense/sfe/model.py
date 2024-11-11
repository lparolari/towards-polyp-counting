import lightning as L
import torch

from polypsense.sfe.backbone import ResNetSimCLR
from polypsense.sfe.simclr import SimCLR


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class MyModel(L.LightningModule):  # TODO: change name
    def __init__(self, args):
        super().__init__()

        if isinstance(args, dict):
            from argparse import Namespace

            args = Namespace(**args)

        self.args = args

        self.backbone = ResNetSimCLR(
            base_model=args.backbone_arch,
            pretrained=args.backbone_weights,
            out_dim=args.backbone_out_dim,
        )
        self.simclr = SimCLR(n_views=args.n_views, temperature=args.temperature)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters(args)

    def forward(self, images):
        return self.backbone(images)

    def step(self, batch, stage=None):
        features = self(batch)
        logits, labels = self.simclr(features)

        loss = self.criterion(logits, labels)

        if stage:
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc_top1", top1)
            self.log(f"{stage}_acc_top5", top5)

        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="test")

    def configure_optimizers(self):
        lr = self.args.lr * self.args.batch_size / 256
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
