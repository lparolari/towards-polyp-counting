import os

import lightning as L

from polypsense.mve.alpha import UpdateAlphaCallback
from polypsense.mve.dm import SequenceTemporalDataModule
from polypsense.mve.model import MultiViewEncoder
from polypsense.sfe.model import MyModel as SingleFrameEncoder


def train(args):
    L.seed_everything(args.seed, workers=True)

    dm = get_datamodule(args)
    model = get_model(args)

    trainer = L.Trainer(
        logger=get_logger(args),
        callbacks=get_callbacks(args),
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator="cuda",
        log_every_n_steps=1,
    )

    trainer.test(model, dm)
    trainer.fit(model, dm, ckpt_path=args.resume and args.ckpt_path)
    trainer.test(model, dm, ckpt_path="best")


# def eval(args):
#     L.seed_everything(args.seed, workers=True)

#     dm = get_datamodule(args)
#     model = MyModel.load_from_checkpoint(args.ckpt_path)

#     trainer = L.Trainer(
#         accelerator=args.accelerator,
#         logger=get_logger(args),
#         log_every_n_steps=1,
#     )

#     trainer.test(model, dm)


def get_datamodule(args):
    return SequenceTemporalDataModule(
        dataset_root=args.dataset_root,
        s=args.s,
        im_size=args.im_size,
        aug_vflip=args.aug_vflip,
        aug_hflip=args.aug_hflip,
        aug_affine=args.aug_affine,
        aug_colorjitter=args.aug_colorjitter,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        alpha=args.alpha_initial_value,
    )


def get_model(args):
    sfe = SingleFrameEncoder.load_from_checkpoint(
        # "/home/lparolar/Projects/polypsense/wandb_logs/simclr/oex79j1n/checkpoints/epoch=8-step=28980-val_acc_top1=59.83.ckpt",  # sr202
        args.sfe_ckpt,
        map_location="cuda",
    )

    # in the paper they actually do not freeze the model, but we can allow it
    if args.sfe_freeze:
        sfe.eval()
        for param in sfe.parameters():
            param.requires_grad = False

    return MultiViewEncoder(
        sfe,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        lr=args.lr,
    )


def get_logger(args):
    logger = L.pytorch.loggers.WandbLogger(
        entity="lparolari",
        project="polypsense-mve",
        id=args.exp_id,
        name=args.exp_name,
        save_dir=os.path.join(os.getcwd(), "wandb_logs"),
        allow_val_change=True,
        resume="allow",
    )
    logger.experiment.config.update(args, allow_val_change=True)
    return logger


def get_callbacks(args):
    return [
        # save best ckpt
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{step}-{val_loss:.2f}",
        ),
        # save last ckpt (for resuming)
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="step",
            mode="max",
            filename="{epoch}-{step}",
        ),
        L.pytorch.callbacks.LearningRateMonitor(),
        # UpdateAlphaCallback(),
    ] + (
        [
            UpdateAlphaCallback(
                initial_alpha=args.alpha_initial_value,
                min_alpha=args.alpha_min_value,
                max_epochs=args.alpha_max_epochs,
            )
        ]
        if args.alpha_scheduler in ["epoch"]
        else []
    )
