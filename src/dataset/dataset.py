import os
import glob
import torchaudio
from torch.utils.data import Dataset


class DataModule:
    def __init__(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class ToyDataset(Dataset):
    def __init__(self, data_path: str, mode: str = "binaural", *args, **kwargs):
        super().__init__()
        assert mode in [
            "binaural",
            "monaural",
            "both",
        ], "`mode` parameters must be binaural, monaural, or both"
        self.data_path = data_path
        self.mode = mode

        self.clean_binaural_path = os.path.join(self.data_path, f"clean_bi")
        self.clean_monaural_path = os.path.join(self.data_path, f"clean_mono")
        self.noisy_path = os.path.join(self.data_path, "noisy")

        # A list of tuple, containing a pair of noisy and clean filename. In tuple, the first and second entries are noisy and clean filenames, respectively.
        self.data = []
        clean_binaural_filenames_path = glob.glob(os.path.join(self.clean_binaural_path, "*.wav"))
        for clean_filename in clean_binaural_filenames_path:
            clean_filename = os.path.split(clean_filename)[-1]
            clean_filename_wo_extension = os.path.splitext(clean_filename)[0]
            clean_binaural_filename_path = os.path.join(self.clean_binaural_path, clean_filename)
            clean_monaural_filename_path = os.path.join(self.clean_monaural_path, clean_filename)

            noisy_filenames = glob.glob(
                os.path.join(self.noisy_path, f"{clean_filename_wo_extension}_*")
            )
            for noisy_filename in noisy_filenames:
                self.data.append(
                    (noisy_filename, clean_binaural_filename_path, clean_monaural_filename_path)
                )

    def __getitem__(self, index):
        noisy_filename, clean_binaural_filename, clean_monaural_filename = self.data[index]
        noisy_utterance, _ = torchaudio.load(noisy_filename)
        # clean_binaural_utterance, _ = torchaudio.load(clean_binaural_filename)
        clean_monaural_utterance, _ = torchaudio.load(clean_monaural_filename)
        clean_binaural_utterance = clean_monaural_utterance.repeat(2, 1)

        if self.mode == "binaural":
            return noisy_utterance, clean_binaural_utterance
        elif self.mode == "monaural":
            return noisy_utterance, clean_monaural_utterance
        elif self.mode == "both":
            return noisy_utterance, clean_binaural_utterance, clean_monaural_utterance

    def __len__(self):
        return len(self.data)
