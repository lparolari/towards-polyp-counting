import os

import lightning as L

from polypsense.sfe.dm import TemporalDataModule, AugmentationDataModule
from polypsense.sfe.model import MyModel


def train(args):
    assert args.n_views == 2, "Only 2 views are supported."

    L.seed_everything(args.seed, workers=True)

    dm = get_datamodule(args)
    model = MyModel(args)

    trainer = L.Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        logger=get_logger(args),
        callbacks=get_callbacks(args),
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1,  # we want validation set to use the same random sequence for every epoch
    )

    trainer.test(model, dm)
    trainer.fit(model, dm, ckpt_path=args.resume and args.ckpt_path)
    trainer.test(model, dm, ckpt_path="best")


def eval(args):
    L.seed_everything(args.seed, workers=True)

    dm = get_datamodule(args)
    model = MyModel.load_from_checkpoint(args.ckpt_path)

    trainer = L.Trainer(
        accelerator=args.accelerator,
        logger=get_logger(args),
        log_every_n_steps=1,
    )

    trainer.test(model, dm)


def get_datamodule(args):
    contrastive_dms = {
        "temporal": lambda: TemporalDataModule(args),
        "augmentation": lambda: AugmentationDataModule(args),
    }
    return contrastive_dms[args.contrastive_strategy]()


def get_logger(args):
    logger = L.pytorch.loggers.WandbLogger(
        project="simclr",
        id=args.exp_id,
        name=args.exp_name,
        notes=args.exp_notes,
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
            monitor="val_acc_top1",
            mode="max",
            filename="{epoch}-{step}-{val_acc_top1:.2f}",
            save_last="link",
        ),
        # save last ckpt (for resuming)
        L.pytorch.callbacks.ModelCheckpoint(
            monitor="step",
            mode="max",
            filename="{epoch}-{step}",
        ),
        L.pytorch.callbacks.LearningRateMonitor(),
    ] + (
        [
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_acc_top1",
                mode="max",
                patience=args.patience,
            )
        ]
        if args.patience
        else []
    )
