import os
import glob
import torchaudio
from torch.utils.data import Dataset

MODE_TO_PATH_NAME = {"binaural": "bi", "monaural": "mono"}


class ToyDataset(Dataset):
    def __init__(self, data_path: str, mode: str = "binaural"):
        super().__init__()
        assert mode in ["binaural", "monaural"], "`mode` parameters must be binaural or monaural"
        self.data_path = data_path
        self.mode = MODE_TO_PATH_NAME[mode]

        self.clean_path = os.path.join(self.data_path, f"clean_{self.mode}")
        self.noisy_path = os.path.join(self.data_path, "noisy")

        # A list of tuple, containing a pair of noisy and clean filename. In tuple, the first and second entries are noisy and clean filenames, respectively.
        self.data = []
        for clean_filename in glob.glob(os.path.join(self.clean_path, "*.wav")):
            clean_filename_wo_extension = os.path.split(clean_filename)[-1]
            clean_filename_wo_extension = os.path.splitext(clean_filename_wo_extension)[0]

            noisy_filenames = glob.glob(
                os.path.join(self.noisy_path, f"{clean_filename_wo_extension}_*")
            )
            for noisy_filename in noisy_filenames:
                self.data.append((noisy_filename, clean_filename))

    def __getitem__(self, index):
        noisy_filename, clean_filename = self.data[index]
        noisy_utterance, _ = torchaudio.load(noisy_filename)
        clean_utterance, _ = torchaudio.load(clean_filename)

        return noisy_utterance, clean_utterance

    def __len__(self):
        return len(self.data)
