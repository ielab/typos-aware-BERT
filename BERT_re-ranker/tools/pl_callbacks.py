import math
from typing import Sequence, Union

from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torch.nn import Module


class CheckpointEveryEpoch(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
            self,
            start_epoch,
            max_epoch,
            gap,
            file_path,
    ):

        self.check_steps = list(range(start_epoch-1, max_epoch, gap))
        print("Checkpoints will be saved at end of epochs of", self.check_steps)
        self.file_path = file_path

    def on_epoch_end(self, trainer: Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        if trainer.current_epoch in self.check_steps:
            ckpt_path = f"{self.file_path}_epoch{trainer.current_epoch+1}.ckpt"
            trainer.save_checkpoint(ckpt_path)


class CheckpointEveryNSteps(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        file_path,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            file_path: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        print("Checkpoint save frequency:", save_step_frequency)
        self.save_step_frequency = save_step_frequency
        self.file_path = file_path

    def on_batch_end(self, trainer: Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0 and global_step > 0:
            ckpt_path = f"{self.file_path}_step{global_step}.ckpt"
            trainer.save_checkpoint(ckpt_path)


class BYOLMAWeightUpdate(Callback):
    """
    Weight update rule from BYOL.
    Your model should have:
        - ``self.online_network``
        - ``self.target_network``
    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[BYOLMAWeightUpdate()])
    """

    def __init__(self, initial_tau: float = 0.996):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.online_model
        target_net = pl_module.target_model

        # update weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: LightningModule, trainer: Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs  # type: ignore[attr-defined]
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, online_net: Union[Module, Tensor], target_net: Union[Module, Tensor]) -> None:
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(
            online_net.named_parameters(),  # type: ignore[union-attr]
            target_net.named_parameters()  # type: ignore[union-attr]
        ):
            if 'weight' in name:
                target_p.data = self.current_tau * target_p.data + (1 - self.current_tau) * online_p.data