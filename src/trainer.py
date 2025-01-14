import torch
from typing import List
from torch.optim.lr_scheduler import StepLR
from .callbacks import Callback


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    def __init__(
        self,
        model,
        data_module,
        optimizer,
        scheduler=None,
        callbacks: List[Callback] = None,
        device="cuda",
    ):
        self.model = model
        self.data_module = data_module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks if callbacks is not None else []
        self.device = device
        self.model.to(device)

    def _call_callbacks(self, hook_name, **kwargs):
        for callback in self.callbacks:
            hook = getattr(callback, hook_name, None)
            if callable(hook):
                hook(self, **kwargs)

    def fit(self, epochs):
        self._call_callbacks("on_start")
        for epoch in range(1, epochs + 1):
            self.training_loop(epoch)
            self.validation_loop(epoch)
        self._call_callbacks("on_end")

    def training_loop(self, epoch):
        self._call_callbacks("on_train_epoch_start", epoch=epoch)
        self.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(self.data_module.train_dataloader(), 1):
            self._call_callbacks("on_train_batch_start", batch=batch, batch_idx=batch_idx)
            output, loss = self.training_step(batch)
            epoch_loss += loss.item()
            self._call_callbacks(
                "on_train_batch_end", batch=batch, batch_idx=batch_idx, output=output, loss=loss
            )
            if self.scheduler:
                self.scheduler.step()
        average_loss = epoch_loss / len(self.data_module.train_dataloader())
        self._call_callbacks("on_train_epoch_end", epoch=epoch, logs={"loss": average_loss})

    def training_step(self, batch):
        self.optimizer.zero_grad()
        output, loss = self.model.step(batch)
        loss.backward()
        self.optimizer_step()
        return output, loss

    def validation_loop(self, epoch):
        self._call_callbacks("on_validation_epoch_start", epoch=epoch)
        self.eval()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(self.data_module.val_dataloader(), 1):
            self._call_callbacks("on_validation_batch_start", batch=batch, batch_idx=batch_idx)
            output, loss = self.validation_step(batch)
            epoch_loss += loss.item()
            self._call_callbacks(
                "on_validation_batch_end",
                batch=batch,
                batch_idx=batch_idx,
                output=output,
                loss=loss,
            )
        average_loss = epoch_loss / len(self.data_module.val_dataloader())
        self._call_callbacks("on_validation_epoch_end", epoch=epoch, logs={"loss": average_loss})

    def validation_step(self, batch):
        with torch.no_grad():
            output, loss = self.model.step(batch)
        return output, loss

    def optimizer_step(self, *args, **kwargs):
        self.optimizer.step()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, save_path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            },
            save_path,
        )

    def load(self, load_path):
        state_dict = torch.load(load_path)
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict.get("scheduler", None))
