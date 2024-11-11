import lightning as L


def get_alpha(initial_alpha, min_alpha, progress):
    return min_alpha + (initial_alpha - min_alpha) * (1 - progress)


class UpdateAlphaCallback(L.Callback):
    def __init__(self, initial_alpha, min_alpha, max_epochs):
        """
        Args:
            initial_alpha: initial value of alpha in [0, 1]
            min_alpha: minimum value of alpha in [0, 1], reached at max_epochs. Must be <= initial_alpha
            max_epochs: number of epochs to reach min_alpha
        """
        if not 0 <= initial_alpha <= 1:
            raise ValueError("initial_alpha must be in [0, 1]")

        if not 0 <= min_alpha <= 1:
            raise ValueError("min_alpha must be in [0, 1]")

        if min_alpha > initial_alpha:
            raise ValueError("min_alpha must be <= initial_alpha")

        if max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")

        self.initial_alpha = initial_alpha
        self.min_alpha = min_alpha
        self.max_epochs = max_epochs

    def on_train_epoch_start(self, trainer, pl_module):

        current_epoch = trainer.current_epoch + 1

        # starts from 0, ends at 1 when max_epochs is reached and stays at 1
        progress = min(current_epoch / self.max_epochs, 1)

        alpha = get_alpha(self.initial_alpha, self.min_alpha, progress)

        # set alpha in the training dataset
        trainer.datamodule.train_dataset.alpha = alpha

        # log alpha
        pl_module.log("alpha", alpha)
