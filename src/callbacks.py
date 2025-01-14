import os
import yaml
import sys
import wandb
import torch
import numpy as np
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from .utils.other import get_lr, si_sdr


class Callback:
    def on_start(self, trainer):
        """Called at the beginning of training"""
        pass

    def on_train_epoch_start(self, trainer, epoch):
        """Called at the beginning of each training epoch"""
        pass

    def on_train_batch_start(self, trainer, batch, batch_idx):
        """Called at the beginning of each training batch"""
        pass

    def on_train_batch_end(self, trainer, batch, batch_idx, output, loss):
        """Called at the end of each training batch"""
        pass

    def on_train_epoch_end(self, trainer, epoch, logs=None):
        """Called at the end of each training epoch"""
        pass

    def on_validation_epoch_start(self, trainer, epoch):
        """Called at the beginning of validation epoch"""
        pass

    def on_validation_batch_start(self, trainer, batch, batch_idx):
        """Called at the beginning of validation batch"""
        pass

    def on_validation_batch_end(self, trainer, batch, batch_idx, output, loss):
        """Called at the end of validation batch"""
        pass

    def on_validation_epoch_end(self, trainer, epoch, logs=None):
        """Called at the end of each training epoch"""
        pass

    def on_end(self, trainer):
        """Called at the end of validation"""
        pass


class TQDMProgressBar(Callback):
    def __init__(self):
        self.pbar = None
        self.training_loss = None
        self.validation_loss = None

    def on_train_batch_start(self, trainer, batch, batch_idx):
        batch_length = len(trainer.data_module.train_dataloader())
        if self.pbar is None:
            self.pbar = tqdm(total=batch_length, desc=f"Training")

    def on_train_batch_end(self, trainer, batch, batch_idx, output, loss):
        self.pbar.update(1)
        self.pbar.set_postfix({"loss": loss.item()})

    def on_train_epoch_end(self, trainer, epoch, logs=None):
        self.pbar.close()
        self.pbar = None
        self.training_loss = logs.get("loss")

    def on_validation_batch_start(self, trainer, batch, batch_idx):
        batch_length = len(trainer.data_module.val_dataloader())
        if self.pbar is None:
            self.pbar = tqdm(total=batch_length, desc=f"Validation")

    def on_validation_batch_end(self, trainer, batch, batch_idx, output, loss):
        self.pbar.update(1)

    def on_validation_epoch_end(self, trainer, epoch, logs=None):
        self.pbar.close()
        self.pbar = None
        self.validation_loss = logs.get("loss")
        print(
            f"Epoch {epoch} ended with loss: Training loss -> {self.training_loss:.4f} Validation loss -> {self.validation_loss:.4f}"
        )


class WanDBLogger(Callback):
    def __init__(self, wandb_config):
        self.wandb_config = wandb_config
        self.output_path = os.path.join(self.wandb_config["config"]["output_path"], "weights")
        os.makedirs(self.output_path)

    def on_start(self, trainer):
        wandb.init(**self.wandb_config)

    def on_train_epoch_start(self, trainer, epoch):
        wandb.log({"epoch": epoch, "lr": get_lr(trainer.optimizer)})

    def on_train_epoch_end(self, trainer, epoch, logs=None):
        wandb.log({"epoch": epoch, "train_loss": logs.get("loss")})

    def on_validation_epoch_end(self, trainer, epoch, logs=None):
        wandb.log({"epoch": epoch, "val_loss": logs.get("loss")})

        save_path = os.path.join(self.output_path, f"epoch_{epoch}.ckpt")
        trainer.save(save_path)

    # def on_end(self, trainer):
    #     wandb.finish(exit_code=0)


class ValidationInference(Callback):
    def __init__(
        self,
        save_path,
    ):
        self.output = []
        self.target = []
        self.save_path = save_path
        self.best_pesq = None
        self.best_epoch = None

    def on_validation_epoch_start(self, trainer, epoch):
        # TODO: clear list of output and label
        self.output = []
        self.target = []

    def on_validation_batch_end(self, trainer, batch, batch_idx, output, loss):
        # TODO: append minibatch output to self.output, do the same thing to batch_label
        clean_mono_utterance = batch[-1]

        self.output.append(output.detach())
        self.target.append(clean_mono_utterance.detach())

    def on_validation_epoch_end(self, trainer, epoch, logs=None):
        # TODO: Calculate other metrics, including accuracy, eer, min_dcf, and print or wandb
        self.output = torch.cat(self.output, 0).data.cpu().numpy()
        self.target = torch.cat(self.target, 0).squeeze(1).data.cpu().numpy()

        self.output = np.mean(self.output, axis=1)

        n_utterance, _ = self.output.shape

        _pesq, _si_sdr, _estoi = 0.0, 0.0, 0.0
        for index in tqdm(range(n_utterance), desc="Calculate Metrics"):
            _si_sdr += si_sdr(self.target[index], self.output[index])
            _pesq += pesq(16000, self.target[index], self.output[index], "wb")
            _estoi += stoi(self.target[index], self.output[index], 16000, extended=True)

        _si_sdr /= n_utterance
        _pesq /= n_utterance
        _estoi /= n_utterance

        if self.best_pesq is None or _pesq > self.best_pesq:
            self.best_epoch = epoch
            self.best_pesq = _pesq

        wandb.log(
            {
                "epoch": epoch,
                "valid_si_sdr": _si_sdr,
                "valid_pesq": _pesq,
                "vaild_estoi": _estoi,
            }
        )

    def on_end(self, trainer):
        # TODO Load Best model and Inference on the test-set.
        wandb.log({"best_epoch": self.best_epoch, "best_pesq": self.best_pesq})
