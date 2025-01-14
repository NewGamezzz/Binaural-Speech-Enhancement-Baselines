import torch
import soundfile as sf

from src.backbones.BCCTN import BinauralAttentionDCNN
from src.dataset import ToyDataset

from torch.utils.data import DataLoader

model = BinauralAttentionDCNN()

data_path = "../test"
dataset = ToyDataset(data_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in dataloader:
    noisy_utterance, clean_utterance = data
    batch_size = noisy_utterance.shape[0]
    print(noisy_utterance.shape)
    print(clean_utterance.shape)
    for batch_index in range(batch_size):
        print(clean_utterance[batch_index].shape)
        sf.write(
            f"./test/clean/{batch_index}.wav",
            torch.transpose(clean_utterance[batch_index], 0, 1).numpy(),
            16000,
            "PCM_24",
        )
        sf.write(
            f"./test/noisy/{batch_index}.wav",
            torch.transpose(noisy_utterance[batch_index], 0, 1).numpy(),
            16000,
            "PCM_24",
        )
        print()
    break
