import torch
import soundfile as sf
import numpy as np

from src.backbones.BCCTN import BinauralAttentionDCNN
from src.losses import BinauralLoss
from src.dataset import ToyDataset

from torch.utils.data import DataLoader

model = BinauralAttentionDCNN()

data_path = "../test"
dataset = ToyDataset(data_path, mode="both")
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

loss_func = BinauralLoss()
for data in dataloader:
    noisy_utterance, clean_binaural_utterance, clean_monaural_utterance = data
    print(clean_binaural_utterance.shape)

    out = model(noisy_utterance)
    print(out.shape)
    loss = loss_func(out, clean_binaural_utterance)
    print(loss.shape)
    print(clean_monaural_utterance.squeeze(1).shape)

    out = out.data.cpu().numpy()
    print(out.shape, np.mean(out, axis=1).shape)

    # for i in range(clean_monaural_utterance.shape[0]):
    #     sf.write(
    #         f"./test/clean_mono/{i}.wav",
    #         torch.transpose(clean_monaural_utterance[i], 0, 1).numpy(),
    #         16000,
    #         "PCM_24",
    #     )
    #     sf.write(
    #         f"./test/clean_bi/{i}.wav",
    #         torch.transpose(clean_binaural_utterance[i], 0, 1).numpy(),
    #         16000,
    #         "PCM_24",
    #     )
    #     sf.write(
    #         f"./test/noisy/{i}.wav",
    #         torch.transpose(noisy_utterance[i], 0, 1).numpy(),
    #         16000,
    #         "PCM_24",
    #     )
    break
