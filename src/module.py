import torch.nn as nn


class BinauralSpeechEnhancement(nn.Module):
    def __init__(self, model, loss_func, device="cuda"):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.device = device
        self.to(device)

    def forward(self, noisy_utterance):
        noisy_utterance = noisy_utterance.to(self.device)

        return self.model(noisy_utterance)

    def step(self, batch):
        noisy_utterance, clean_utterance, _ = batch
        noisy_utterance, clean_utterance = noisy_utterance.to(self.device), clean_utterance.to(
            self.device
        )

        denoised_utterance = self.model(noisy_utterance)
        loss = self.loss_func(denoised_utterance, clean_utterance)
        return denoised_utterance, loss
