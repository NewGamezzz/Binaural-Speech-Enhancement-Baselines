import os
import yaml
import torch
import argparse

from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from src import factory
from src.utils.other import get_lr, si_sdr


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weight_path", required=True, help="Directory of the weights."
    )
    parser.add_argument("--weight_epoch", required=True, help="An epoch to be loaded.")
    parser.add_argument("--device", default="cuda", help="Inference on cpu or cuda")

    args = parser.parse_args()
    return args


def load_yaml(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def inference(data_loader, model, device="cuda"):
    # TODO: Implement inference process and calculate metrics.
    output_utterances = []
    target_utterances = []
    for data in data_loader:
        noisy_utterance, clean_binaural_utterance, clean_monaural_utterance = data
        noisy_utterance, clean_binaural_utterance, clean_monaural_utterance = (
            noisy_utterance.to(device),
            clean_binaural_utterance.to(device),
            clean_monaural_utterance.to(device),
        )
        output = model(noisy_utterance)
        output_utterances.append(output)
        target_utterances.append(clean_monaural_utterance)

    output_utterances = torch.cat(output_utterances, dim=0).data.cpu().numpy()
    target_utterances = (
        torch.cat(target_utterances, dim=0).squeeze(1).data.cpu().numpy()
    )
    n_utterance, _ = output.shape

    _pesq, _si_sdr, _estoi = 0.0, 0.0, 0.0
    for index in tqdm(range(n_utterance), desc="Calculate Metrics"):
        _si_sdr += si_sdr(target_utterances[index], output_utterances[index])
        _pesq += pesq(16000, target_utterances[index], output_utterances[index], "wb")
        _estoi += stoi(
            target_utterances[index], output_utterances[index], 16000, extended=True
        )

    _si_sdr /= n_utterance
    _pesq /= n_utterance
    _estoi /= n_utterance
    print("SI-SDR:", _si_sdr, "PESQ:", _pesq, "ESTOI:", _estoi)


if __name__ == "__main__":
    args = get_args()
    print(args)

    state_dict_path = os.path.join(
        args.weight_path, f"weight/epoch_{args.weight_epoch}.ckpt"
    )
    config_path = os.path.join(args.weight_path, ".hydra.yaml")

    state_dict = torch.load(state_dict_path)
    config = load_yaml(config_path)

    data_config = config.dataset
    data_module = factory.create_data_module(data_config)

    model_config = config.model
    model = factory.create_model(model_config)
    model.load_state_dict(state_dict["model"])
    model.to(args.device)
    model.eval()

    inference(data_module.test_dataloader(), model, device=args.device)
