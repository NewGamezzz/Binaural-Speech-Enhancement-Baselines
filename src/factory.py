import os
import copy
import torch
import hydra
import wandb
from torch.utils.data import DataLoader
from .backbones import BCCTN
from .dataset import DataModule, ToyDataset
from .losses import BinauralLoss
from .module import BinauralSpeechEnhancement
from .callbacks import TQDMProgressBar, WanDBLogger, ValidationInference


def create_data_module(config):
    train_path = config.pop("train_path")
    val_path = config.pop("val_path")
    test_path = config.pop("test_path")

    train_dataset = ToyDataset(**dict(config, data_path=train_path))
    val_dataset = ToyDataset(**dict(config, data_path=val_path))
    test_dataset = ToyDataset(**dict(config, data_path=test_path))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 1),
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 1),
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", 1),
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
        shuffle=True,
    )
    data_module = DataModule(train_loader, val_loader, test_loader)
    return data_module


def create_model(config):
    name_to_model = {"BCCTN": BCCTN.BinauralAttentionDCNN}
    name_to_loss = {"binaural_loss": BinauralLoss}
    name = config.pop("name")
    loss_config = config.pop("loss")
    loss_name = loss_config.pop("name")
    device = config.pop("device", "cuda")

    model = name_to_model[name](**config)
    loss_func = name_to_loss[loss_name](**loss_config)
    module = BinauralSpeechEnhancement(model, loss_func, device)
    return module


def create_callback(config, **kwargs):
    callbacks_names_to_func = {
        "tqdm": create_tqdm_callback,
        "wandb": create_wandb_callback,
        "validation_inference": create_validation_inference_callback,
    }
    name = config.pop("name")
    return callbacks_names_to_func[name](config=config, **kwargs)


def create_tqdm_callback(*args, **kwargs):
    return TQDMProgressBar()


def create_wandb_callback(config, general_config, *args, **kwargs):
    wandb_token = config.pop("WANDB_TOKEN")
    wandb_config = config.pop("WANDB_INIT")
    wandb.login(key=wandb_token)
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    wandb_name = "/".join(output_path.split("/")[-2:])
    general_config["output_path"] = output_path

    # Check wandb callback and pop
    wandb_idx = -1
    for idx, callback in enumerate(
        general_config["trainer"]["callbacks"]
    ):  # There must be callback in the config to call this function
        if callback["name"] == "wandb":
            wandb_idx = idx
    if wandb_idx != -1:
        general_config["trainer"]["callbacks"].pop(wandb_idx)

    wandb_config["config"] = general_config
    wandb_config["name"] = wandb_name
    return WanDBLogger(wandb_config)


def create_validation_inference_callback(config, diffusion, data_module, *args, **kwargs):
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    folder_name = config["save_path"].replace("./", "")
    config["save_path"] = os.path.join(output_path, folder_name)

    return ValidationInference(**config)


def create_trainer(diffusion, data_module, config):
    general_config = copy.deepcopy(config)
    trainer_config = config["trainer"]

    optimizer_config = trainer_config.pop("optimizer")
    optimizer = torch.optim.Adam(diffusion.parameters(), **optimizer_config)

    kwargs = {"diffusion": diffusion, "data_module": data_module, "general_config": general_config}
    callbacks = []
    callbacks_config = trainer_config.get("callbacks", [])
    for callback_config in callbacks_config:
        callback = create_callback(callback_config, **kwargs)
        callbacks.append(callback)

    trainer = Trainer(diffusion, data_module, optimizer, callbacks=callbacks, scheduler=scheduler)

    return trainer
