import os
import yaml
import sys
import wandb
import torch
import numpy as np
from tqdm import tqdm
from .utils.other import get_lr


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


class LogSpoofMetrics(Callback):
    def __init__(self, output_path, enhance=None, mask=False):
        self.output = []
        self.label = []
        self.mask = mask
        self.enhance = enhance
        self.output_path = os.path.join(output_path, "weights")
        self.best_eer = None
        self.best_epoch = None

    def on_train_epoch_start(self, trainer, epoch):
        self.output = []
        self.label = []

    def on_train_batch_end(self, trainer, batch, batch_idx, output, loss):
        # TODO: append minibatch output to self.output, do the same thing to batch_label
        minibatch_label = batch[1]

        self.output.append(output.detach())
        self.label.append(minibatch_label.detach())

    def on_train_epoch_end(self, trainer, epoch, logs=None):
        # TODO: Calculate other metrics, including accuracy, eer, min_dcf, and print or wandb
        self.output = torch.cat(self.output, 0)
        self.label = torch.cat(self.label, 0).data.cpu().numpy()
        prob = torch.nn.functional.softmax(self.output, dim=1)[:, 1].cpu().numpy()

        # Calculate accuracy
        _, pred = self.output.max(dim=1)
        num_correct = (pred.cpu().numpy() == self.label).sum(axis=0).item()
        num_total = self.output.size(0)
        accuracy = (num_correct / num_total) * 100

        # Calculate eer, mindcf
        eer = compute_eer(prob[self.label == 1], prob[self.label == 0])[0] * 100

        cm_keys = np.where(self.label == 1, "bonafide", "spoof")
        minDCF, _, _, _ = calculate_minDCF_EER_CLLR_actDCF(
            cm_scores=prob, cm_keys=cm_keys, output_file="./tmp.txt"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_accuracy": accuracy,
                "train_eer": eer,
                "train_minDCF": minDCF,
            }
        )

    def on_validation_epoch_start(self, trainer, epoch):
        # TODO: clear list of output and label
        self.output = []
        self.label = []

    def on_validation_batch_end(self, trainer, batch, batch_idx, output, loss):
        # TODO: append minibatch output to self.output, do the same thing to batch_label
        minibatch_label = batch[1]

        self.output.append(output.detach())
        self.label.append(minibatch_label.detach())

    def on_validation_epoch_end(self, trainer, epoch, logs=None):
        # TODO: Calculate other metrics, including accuracy, eer, min_dcf, and print or wandb
        self.output = torch.cat(self.output, 0)
        self.label = torch.cat(self.label, 0).data.cpu().numpy()
        prob = torch.nn.functional.softmax(self.output, dim=1)[:, 1].cpu().numpy()

        # Calculate accuracy
        _, pred = self.output.max(dim=1)
        num_correct = (pred.cpu().numpy() == self.label).sum(axis=0).item()
        num_total = self.output.size(0)
        accuracy = (num_correct / num_total) * 100

        # Calculate eer, mindcf
        eer = compute_eer(prob[self.label == 1], prob[self.label == 0])[0] * 100

        cm_keys = np.where(self.label == 1, "bonafide", "spoof")
        minDCF, _, _, _ = calculate_minDCF_EER_CLLR_actDCF(
            cm_scores=prob, cm_keys=cm_keys, output_file="./tmp.txt"
        )
        if self.best_eer is None or self.best_eer > eer:
            self.best_eer = eer
            self.best_epoch = epoch

        wandb.log(
            {
                "epoch": epoch,
                "valid_accuracy": accuracy,
                "valid_eer": eer,
                "valid_minDCF": minDCF,
            }
        )

    def on_end(self, trainer):
        # TODO Load Best model and Inference on the test-set.
        pass
